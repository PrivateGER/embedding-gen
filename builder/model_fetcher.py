# builder/model_fetcher.py
import open_clip
import torch
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch.models.mtcnn import MTCNN

device = "cuda" if torch.cuda.is_available() else "cpu"

open_clip.download_pretrained({
    "download_hf_hub": 'hf-hub:google/vit-huge-patch14-224-in21k',
}, True)

tokenizer = open_clip.get_tokenizer('hf-hub:google/vit-huge-patch14-224-in21k')

del open_clip
del tokenizer

mtcnn = MTCNN(keep_all=True, device=device)  # For face detection
del mtcnn

facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
del facenet
