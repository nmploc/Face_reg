#!/usr/bin/env python3
"""
face_demo_gpu_dml.py
Realtime face recognition demo on AMD RX 6600M using DirectML.
- Prefer DirectML (torch-directml) for AMD/Windows GPU
- Else CPU

Usage:
  python face_demo_gpu_dml.py
Requirements:
  pip install torch torchvision facenet-pytorch opencv-python numpy
  pip install torch-directml   # for AMD GPU (RX 6600M)
"""

import os
import sys
from pathlib import Path
import time
import numpy as np
import cv2
from PIL import Image

# Torch import
try:
    import torch
except Exception as e:
    print(f"[ERROR] Cannot import torch: {e}")
    sys.exit(1)

# Force DirectML first
def get_compute_device():
    try:
        import torch_directml
        dml_dev = torch_directml.device()
        print("[INFO] Using DirectML device (AMD RX 6600M).")
        return dml_dev
    except Exception as e:
        print(f"[WARN] DirectML not available ({e}) → fallback to CPU.")
        return torch.device("cpu")

DEVICE = get_compute_device()

# facenet-pytorch
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
except Exception as e:
    print(f"[ERROR] Cannot import facenet-pytorch: {e}")
    print("Install with: pip install facenet-pytorch")
    sys.exit(1)

# Config
GALLERY_DIR = "gallery"
WEBCAM_ID = 0
FRAME_RESIZE = (640, 480)
MATCH_THRESHOLD = 0.55
IMAGE_SIZE = 160

def build_models(device):
    # Force MTCNN to run on CPU to avoid DirectML tensor rank issues
    mtcnn = MTCNN(
        image_size=IMAGE_SIZE,
        margin=14,
        keep_all=True,
        device=torch.device("cpu"),
        weights_only=True  # Add weights_only to avoid warnings
    )
    
    # Load ResNet with weights_only and move to DirectML device
    resnet = InceptionResnetV1(pretrained='vggface2', weights_only=True).eval()
    try:
        resnet = resnet.to(device)
        print("[INFO] InceptionResnetV1 running on DirectML device")
    except Exception as e:
        print(f"[WARNING] Could not move InceptionResnetV1 to DirectML ({device}): {e}")
        print("Falling back to CPU.")
        resnet = resnet.to(torch.device("cpu"))
        device = torch.device("cpu")
    return mtcnn, resnet, device

mtcnn, resnet, DEVICE = build_models(DEVICE)

# Cosine similarity
def cosine_sim_vec(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(b, a)

# Build gallery
def build_gallery_embeddings(gallery_dir=GALLERY_DIR, device=DEVICE):
    gallery_mean = {}
    p = Path(gallery_dir)
    if not p.exists():
        print(f"[WARN] Gallery folder '{gallery_dir}' not found.")
        return gallery_mean
    for img_path in sorted(p.iterdir()):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        name = img_path.stem.split("_")[0]
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot open {img_path}: {e}")
            continue
        try:
            crops = mtcnn(pil)
        except Exception as e:
            print(f"[WARN] MTCNN failed on {img_path}: {e}")
            crops = None
        if crops is None:
            continue

        crops_list = []
        if isinstance(crops, torch.Tensor):
            if crops.dim() == 4:
                for i in range(crops.shape[0]):
                    crops_list.append(crops[i])
            else:
                crops_list.append(crops)
        elif isinstance(crops, list):
            crops_list = crops

        for c in crops_list:
            c = c.unsqueeze(0)
            try:
                c = c.to(device)
            except Exception:
                c = c.to(torch.device("cpu"))
            with torch.no_grad():
                emb = resnet(c)
            emb = emb.cpu().numpy()[0]
            emb = emb / np.linalg.norm(emb)
            gallery_mean.setdefault(name, []).append(emb)
            print(f"Loaded {img_path.name} -> {name}")

    gallery_final = {}
    for name, embs in gallery_mean.items():
        arr = np.vstack(embs)
        mean = np.mean(arr, axis=0)
        mean = mean / np.linalg.norm(mean)
        gallery_final[name] = mean
        print(f"Gallery: {name} ({arr.shape[0]} images)")
    return gallery_final

# Webcam loop
def run_webcam_loop(gallery_embeddings, device=DEVICE):
    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return
    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, FRAME_RESIZE)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        try:
            crops = mtcnn(pil)
        except Exception:
            crops = None
        try:
            boxes, probs = mtcnn.detect(pil)
        except Exception:
            boxes, probs = None, None

        matches = []
        if crops is not None:
            crops_tensors = []
            if isinstance(crops, torch.Tensor):
                if crops.dim() == 4:
                    for i in range(crops.shape[0]):
                        crops_tensors.append(crops[i].unsqueeze(0))
                else:
                    crops_tensors.append(crops.unsqueeze(0))
            elif isinstance(crops, list):
                for c in crops:
                    crops_tensors.append(c.unsqueeze(0))

            for i, c in enumerate(crops_tensors):
                try:
                    c = c.to(device)
                except Exception:
                    c = c.to(torch.device("cpu"))
                with torch.no_grad():
                    emb = resnet(c).cpu().numpy()[0]
                emb = emb / np.linalg.norm(emb)
                best_name, best_score = "Unknown", -1.0
                if gallery_embeddings:
                    names = list(gallery_embeddings.keys())
                    embs = np.vstack([gallery_embeddings[n] for n in names])
                    sims = cosine_sim_vec(emb, embs)
                    idx = int(np.argmax(sims))
                    best_score = float(sims[idx])
                    if best_score >= MATCH_THRESHOLD:
                        best_name = names[idx]
                matches.append((best_name, best_score))

        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(max(0, v)) for v in box]
                label = "Detect"
                if i < len(matches):
                    name, score = matches[i]
                    label = f"{name} ({score:.2f})"
                color = (0,255,0) if "Unknown" not in label else (0,0,255)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, label, (x1, max(12, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Face Demo (DirectML, q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=== Face recognition demo (DirectML RX 6600M) ===")
    gallery = build_gallery_embeddings(GALLERY_DIR, device=DEVICE)
    if not gallery:
        print("[WARN] Gallery empty. Add images to ./gallery and re-run.")
    else:
        print(f"[INFO] Gallery identities: {list(gallery.keys())}")
    run_webcam_loop(gallery, device=DEVICE)
