"""
test_model_load.py
-------------------
Phase 3 — Confirm the trained model loads correctly on your laptop (CPU)

What this does:
1. Rebuilds the EfficientNetB0 architecture (5 output classes)
2. Loads your saved weights (efficientnet_heritage.pth)
3. Runs one dummy inference pass to confirm everything works end-to-end
4. If you point it at a real image, it'll give you an actual prediction

How to run:
    (heritage_env) > cd CV_Model
    (heritage_env) > python test_model_load.py

Expected folder structure (relative to this script):
    heritage-ai/
        CV_Model/
            test_model_load.py   <- this file
        Models/
            efficientnet_heritage.pth
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---- CONFIG ----
MODEL_PATH = os.path.join("..", "Models", "efficientnet_heritage.pth")

# CONFIRMED directly from your Colab notebook's printed output:
#   "Classes found: ['Baroque', 'Gothic', 'Neoclassical', 'Roman', 'Victorian']"
# This is the exact order full_dataset_train.classes produced (alphabetical,
# from ImageFolder), and it's the order the model's output layer was trained
# against — so this list MUST stay in this exact order.
CLASS_NAMES = ["Baroque", "Gothic", "Neoclassical", "Roman", "Victorian"]

# Optional: point this at a real image to get a real prediction instead of a dummy tensor
TEST_IMAGE_PATH = TEST_IMAGE_PATH = r"C:\Users\Amani\Projects\heritage-ai\dataset\Gothic\02_0002.jpg"  # e.g. r"..\Documents\sample_gothic.jpg"


def build_model(num_classes=5):
    """Recreates the EfficientNetB0 architecture with a 5-class output layer,
    matching what you trained in Colab."""
    model = models.efficientnet_b0(weights=None)  # weights=None: we're loading our own
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model


def load_trained_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}\n"
            f"Download efficientnet_heritage.pth from Google Drive into your Models/ folder."
        )

    model = build_model(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()  # inference mode — disables dropout etc.
    return model


def get_transform():
    """Must match the preprocessing used during training (Phase 2):
    224x224 resize + ImageNet normalization (test_transform in your notebook,
    no augmentation — augmentation was training-only)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, dim=0)
    return CLASS_NAMES[predicted_idx.item()], confidence.item(), probabilities


if __name__ == "__main__":
    print("Step 1: Loading model architecture + weights...")
    model = load_trained_model()
    print("  Model loaded successfully.\n")

    transform = get_transform()

    if TEST_IMAGE_PATH and os.path.isfile(TEST_IMAGE_PATH):
        print(f"Step 2: Running inference on real image: {TEST_IMAGE_PATH}")
        image = Image.open(TEST_IMAGE_PATH).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # add batch dimension
    else:
        print("Step 2: No real image provided — running on a dummy random tensor")
        print("        (this only confirms the model architecture/weights load correctly,")
        print("         NOT that predictions are meaningful — set TEST_IMAGE_PATH for that)")
        image_tensor = torch.rand(1, 3, 224, 224)

    predicted_class, confidence, all_probs = predict(model, image_tensor)

    print(f"\n--- Result ---")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nAll class probabilities:")
    for name, prob in zip(CLASS_NAMES, all_probs):
        print(f"  {name}: {prob.item():.2%}")

    print(f"\nIf this ran without errors, your model loads correctly on CPU.")
    print(f"Phase 3 confirmation: DONE.")
