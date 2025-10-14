import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def get_face_embedding(img_path):
    img = Image.open(img_path)
    face = mtcnn(img)
    if face is not None:
        with torch.no_grad():
            embedding = resnet(face.unsqueeze(0))
        return embedding.squeeze().numpy()
    return None

def get_face_embedding_from_frame(frame, box):
    x1, y1, x2, y2 = [int(b) for b in box]

    # Evita indici fuori range
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face_img = frame[y1:y2, x1:x2]

    # Se troppo piccolo o vuoto → skip
    if face_img.size == 0 or (x2 - x1) < 30 or (y2 - y1) < 30:
        return None

    img = Image.fromarray(face_img)

    # 🔒 Proteggi da crash MTCNN
    try:
        face = mtcnn(img)
        if face is not None:
            with torch.no_grad():
                embedding = resnet(face.unsqueeze(0))
            return embedding.squeeze().numpy()
    except Exception:
        return None

    return None


def compare_embeddings(emb1, emb2, threshold=1.0):
    distance = np.linalg.norm(emb1 - emb2)
    return distance < threshold, distance
