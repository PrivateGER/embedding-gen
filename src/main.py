import runpod
import os

import torch
from PIL import Image
import open_clip
from facenet_pytorch import MTCNN, InceptionResnetV1
from runpod.serverless.utils import download_files_from_urls

## load your model(s) into vram here

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    'hf-hub:google/vit-huge-patch14-224-in21k')
tokenizer = open_clip.get_tokenizer('hf-hub:google/vit-huge-patch14-224-in21k')

clip_model = clip_model.to(device)

mtcnn = MTCNN(keep_all=True, device=device)  # For face detection
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def handler(job):
    print(job)

    job_input = job["input"]

    job_input["image_url"] = job_input.get("image_url", "")
    job_input["text"] = job_input.get("text", "")
    job_input["action"] = job_input.get("action", "")

    if job_input["action"] == "generate_image_embeddings":
        if job_input["image_url"] == "":
            return {"error": "image_url is required"}

        image_path = download_files_from_urls(job["id"], job_input["image_url"])[0]
        if image_path is None:
            return {"error": "image_url could not be downloaded"}

        face_embeddings = get_face_embeddings(image_path)
        clip_embedding = get_clip_embedding(image_path)
        os.remove(image_path)

        return {"face_embeddings": face_embeddings.tolist(), "clip_embedding": clip_embedding.tolist()}

    elif job_input["action"] == "generate_text_embeddings":
        if job_input["text"] == "":
            return {"error": "text is required"}

        clip_embedding = get_clip_text_embedding(job_input["text"])

        return {"clip_embedding": clip_embedding.tolist()}


def get_clip_text_embedding(text):
    with torch.no_grad():
        text_features = clip_model.encode_text(tokenizer([text]).to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Convert to numpy and free up memory
        numpy_features = text_features.cpu().detach().numpy().flatten()

        # Explicitly delete tensors
        del text_features

        # Check memory usage and empty cache conditionally
        if torch.cuda.memory_reserved() - torch.cuda.memory_allocated() < 2000 * 1e6:
            torch.cuda.empty_cache()

        return numpy_features


def get_clip_embedding(image_path):
    with torch.no_grad():  # Disable gradient computation
        # Load the image
        image = Image.open(image_path)
        image = preprocess_val(image).unsqueeze(0).to(device)
        image_features = clip_model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Convert to numpy and free up memory
        numpy_features = image_features.cpu().detach().numpy().flatten()

        # Explicitly delete tensors
        del image
        del image_features

        # Check memory usage and empty cache conditionally
        if torch.cuda.memory_reserved() - torch.cuda.memory_allocated() < 2000 * 1e6:  # Less than 100MB free
            torch.cuda.empty_cache()

    return numpy_features


def get_face_embeddings(image_path):
    # Load the image
    image = Image.open(image_path)

    # Remove alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Detect face(s)
    faces = mtcnn(image)

    # If no face is detected, return an empty list
    if faces is None:
        return []

    print(f"Found {len(faces)} faces in {image_path}")

    # Get the embeddings for all detected faces
    with torch.no_grad():
        embeddings = facenet(faces.to(device))

    # Convert to numpy and free up memory
    del faces
    del image

    # Flatten the embeddings
    embeddings = embeddings.cpu().detach().numpy().reshape(embeddings.shape[0], -1)

    # Check memory usage and empty cache conditionally
    if torch.cuda.memory_reserved() - torch.cuda.memory_allocated() < 2000 * 1e6:  # Less than 100MB free
        torch.cuda.empty_cache()

    return embeddings


runpod.serverless.start({
    "handler": handler
})
