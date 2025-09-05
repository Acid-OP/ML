import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# === Paths ===
weights_path = "mangrove_classifier.pth"
img_path = r"G:\Lock in\sih\Testing\mangroove\avicennia_officinalis2.JPG"

# === Transform (same as training) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Model ===
model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
model.load_state_dict(torch.load(weights_path))
model = model.cuda()
model.eval()

# === Load and preprocess image ===
img = Image.open(img_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).cuda()  # add batch dimension

# === Prediction ===
with torch.no_grad():
    output = model(img_tensor)
    pred = torch.argmax(output, 1).item()

# === Mapping index -> label ===
classes = ["mangrove", "non"]
print(f"Prediction: {classes[pred]}")
