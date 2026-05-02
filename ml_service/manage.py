# ml_service/main.py
"""
ML Microservice — DermNet
Charge best_model.pth (EfficientNet-B4) au démarrage et expose POST /predict/
"""
import os
import io
import uuid
import logging
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import boto3
from botocore.client import Config
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import consul

logger = logging.getLogger("ml_service")
logging.basicConfig(level=logging.INFO)

# ─── Configuration ───────────────────────────────────────────────
MODEL_PATH = os.environ.get('MODEL_PATH', '/models/best_model.pth')
MINIO_ENDPOINT = os.environ.get('MINIO_ENDPOINT', 'localhost:9000')
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY', 'dermnet_minio')
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY', 'dermnet_minio_secret')
MINIO_BUCKET = 'images-cheveux'
CONSUL_HOST = os.environ.get('CONSUL_HOST', 'localhost')

# ─── DermNet class labels (23 classes) ──────────────────────────
# ⚠️ À MODIFIER selon votre ordre de classes !
# L'ordre doit correspondre à celui de votre dataset
DERMNET_CLASSES = [
    "Acne and Rosacea",
    "Actinic Keratosis Basal Cell Carcinoma",
    "Atopic Dermatitis",
    "Bullous Disease",
    "Cellulitis Impetigo",
    "Eczema",
    "Exanthems and Drug Eruptions",
    "Hair Loss Alopecia",
    "Herpes HPV and STDs",
    "Light Diseases and Disorders of Pigmentation",
    "Lupus and Connective Tissue diseases",
    "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease",
    "Poison Ivy Plants and Rashes",
    "Psoriasis Lichen Planus",
    "Scabies Lyme Disease and other Infestations",
    "Seborrheic Keratoses and other Benign Tumors",
    "Systemic Disease",
    "Tinea Ringworm Candidiasis",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis Photos",
    "Warts Molluscum and other Viral Infections",
]

# ─── Image preprocessing for EfficientNet-B4 ────────────────────
# 🔥 CORRECTION : EfficientNet-B4 utilise 384x384 (pas 380)
IMG_SIZE = 384

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── Global model state ──────────────────────────────────────────
model_state = {"model": None, "num_classes": len(DERMNET_CLASSES)}


def load_model():
    """
    Charge le modèle EfficientNet-B4 entraîné
    Attention : La structure du modèle doit correspondre à celle utilisée dans l'entraînement
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}\n"
            "→ Copiez best_efficientnet_b4.pth dans ml_service/models/"
        )

    logger.info(f"Chargement du modèle depuis {MODEL_PATH} ...")
    
    # Charger le checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    # Extraire les poids et les métadonnées
    if isinstance(checkpoint, dict):
        # Vérifier les différentes clés possibles
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        num_classes = checkpoint.get('num_classes', len(DERMNET_CLASSES))
    else:
        state_dict = checkpoint
        num_classes = len(DERMNET_CLASSES)

    model_state["num_classes"] = num_classes
    logger.info(f"Nombre de classes détecté : {num_classes}")

    # 🔥 Construire le modèle EfficientNet-B4 avec la même architecture que l'entraînement
    model = build_efficientnet_b4(num_classes)
    
    # Charger les poids
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Clés manquantes dans le checkpoint: {missing_keys[:5]}...")
    if unexpected_keys:
        logger.warning(f"Clés inattendues dans le checkpoint: {unexpected_keys[:5]}...")
    
    model.eval()
    model_state["model"] = model
    
    logger.info(f"✓ Modèle EfficientNet-B4 chargé avec succès — {num_classes} classes")


def build_efficientnet_b4(num_classes: int):
    """
    Construit un modèle EfficientNet-B4 avec la même architecture que l'entraînement
    (identique à celle utilisée dans le notebook Kaggle)
    """
    try:
        from efficientnet_pytorch import EfficientNet
        
        # Charger le modèle pré-entraîné
        model = EfficientNet.from_pretrained('efficientnet-b4')
        
        # 🔥 Remplacer la dernière couche avec la même structure que dans l'entraînement
        # Dans votre notebook, vous avez utilisé :
        # model._fc = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(num_ftrs, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(256, num_classes)
        # )
        
        num_ftrs = model._fc.in_features
        model._fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        logger.info("✓ EfficientNet-B4 chargé avec efficientnet-pytorch")
        return model
        
    except ImportError:
        # Fallback vers torchvision (structure différente, risque de mismatch)
        logger.warning("efficientnet-pytorch non trouvé, utilisation de torchvision")
        from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
        
        model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        
        # Structure différente ! À adapter
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        logger.info("✓ EfficientNet-B4 chargé avec torchvision")
        return model


# ─── MinIO et Consul (inchangés) ─────────────────────────────────
def get_minio_client():
    """Retourne un client MinIO/S3 configuré"""
    return boto3.client(
        's3',
        endpoint_url=f"http://{MINIO_ENDPOINT}",
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1',
    )


def ensure_bucket():
    """Crée le bucket MinIO s'il n'existe pas"""
    try:
        minio = get_minio_client()
        minio.head_bucket(Bucket=MINIO_BUCKET)
        logger.info(f"Bucket '{MINIO_BUCKET}' existe déjà")
    except Exception:
        minio = get_minio_client()
        minio.create_bucket(Bucket=MINIO_BUCKET)
        logger.info(f"Bucket '{MINIO_BUCKET}' créé")


def register_consul():
    """Enregistre le service dans Consul"""
    try:
        c = consul.Consul(host=CONSUL_HOST)
        c.agent.service.register(
            name='ml-service',
            service_id='ml-service-1',
            address='ml_service',
            port=8003,
            check=consul.Check.http('http://ml_service:8003/health', interval='10s'),
        )
        logger.info("✓ Service enregistré dans Consul")
    except Exception as e:
        logger.warning(f"Connexion à Consul impossible : {e}")


# ─── FastAPI lifecycle ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le cycle de vie de l'application"""
    logger.info("Démarrage du service ML...")
    load_model()
    ensure_bucket()
    register_consul()
    logger.info("Service ML prêt")
    yield
    logger.info("Arrêt du service ML")


app = FastAPI(title="DermNet ML Service", version="1.0", lifespan=lifespan)


# ─── Endpoints ──────────────────────────────────────────────────
@app.get("/health")
def health():
    """Endpoint de health check"""
    return {
        "status": "ok",
        "model_loaded": model_state["model"] is not None,
        "model_type": "EfficientNet-B4"
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Prédit la maladie à partir d'une image dermatologique
    """
    model = model_state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    # Lire et prétraiter l'image
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image invalide : {str(e)}")

    # Prétraitement
    tensor = TRANSFORM(pil_image).unsqueeze(0)  # (1, 3, 384, 384)

    # Inférence
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    # Top-5 prédictions
    top_probs, top_indices = torch.topk(probs, k=min(5, len(DERMNET_CLASSES)))
    top_classes = [
        {"label": DERMNET_CLASSES[idx.item()], "score": round(prob.item(), 4)}
        for prob, idx in zip(top_probs, top_indices)
    ]

    maladie = top_classes[0]["label"]
    confidence = top_classes[0]["score"]

    # Sauvegarder l'image dans MinIO (optionnel, non bloquant)
    image_key = f"ml-uploads/{uuid.uuid4()}_{image.filename}"
    try:
        minio = get_minio_client()
        minio.put_object(
            Bucket=MINIO_BUCKET,
            Key=image_key,
            Body=contents,
            ContentType=image.content_type or 'image/jpeg',
        )
        logger.info(f"Image sauvegardée dans MinIO: {image_key}")
    except Exception as e:
        logger.warning(f"Upload MinIO échoué (non bloquant) : {e}")

    return JSONResponse({
        "maladie": maladie,
        "confidence": confidence,
        "top_classes": top_classes,
        "minio_key": image_key,
    })


@app.get("/classes")
def list_classes():
    """Retourne la liste des classes du modèle"""
    return {
        "classes": DERMNET_CLASSES,
        "count": len(DERMNET_CLASSES)
    }


@app.get("/model-info")
def model_info():
    """Retourne des informations sur le modèle chargé"""
    model = model_state["model"]
    return {
        "model_type": "EfficientNet-B4",
        "input_size": IMG_SIZE,
        "num_classes": model_state["num_classes"],
        "is_loaded": model is not None
    }


# ─── Point d'entrée pour exécution directe ──────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
