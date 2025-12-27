import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from src.models.predict_image import predict_image
TEST_IMAGES = [
    "data/processed/ffpp/test/real/amigo1.jpeg",
    "data/processed/ffpp/test/real/amigo2.jpeg",
    "data/processed/ffpp/test/real/amigo3.jpeg",
    "data/processed/ffpp/test/real/amigo4.jpeg",
    "data/processed/ffpp/test/real/amigo5.jpeg",
    "data/processed/ffpp/test/fake/test_fake_1.jpg",
    "data/processed/ffpp/test/fake/test_fake_2.jpg",
    "data/processed/ffpp/test/fake/test_fake_3.jpg",
    "data/processed/ffpp/test/fake/test_fake_4.jpg",
    "data/processed/ffpp/test/fake/test_fake_5.jpg",
]

print("="*60)
print("PRUEBA DE DETECCIÓN DE DEEPFAKES - 10 IMÁGENES DE DEMO")
print("="*60)

results = {"real": [], "fake": []}

for img_path in TEST_IMAGES:
    if not Path(img_path).exists():
        print(f"⚠️ No encontrado: {img_path}")
        continue
    
    pred_label, probs = predict_image(img_path)
    confidence = probs[pred_label]
    
    if pred_label == 0:
        results["real"].append((img_path, confidence))
    else:
        results["fake"].append((img_path, confidence))
    
    emoji = "✅" if (("real" in img_path and pred_label == 0) or 
                    ("fake" in img_path and pred_label == 1)) else "❌"
    
    print(f"{emoji} {Path(img_path).name:25} → {['REAL', 'FAKE'][pred_label]:4} ({confidence:.1%})")

print("\n" + "="*60)
print(f"Predicciones REAL: {len(results['real'])}/5")
print(f"Predicciones FAKE: {len(results['fake'])}/5")
print("="*60)
