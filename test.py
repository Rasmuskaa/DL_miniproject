import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from train import make_model, val_test_transform, device, DATA_DIR, BATCH_SIZE, LEARNING_RATE, SEED
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import random
import numpy as np

# Hvilken model der skal testes (ret disse to for hver model)
MODEL_NAME = "resnet34"
PRETRAINED = True

# Byg stien til checkpointet på samme måde som run_name i train.py
run_name = f"{MODEL_NAME}_{'pretrained' if PRETRAINED else 'FRA_BUNDEN'}_lr{LEARNING_RATE}_seed{SEED}"
CHECKPOINT = f"checkpoints/{run_name}_best.pt"

# Testdata: samme forbehandling som validering, ingen augmentation
test_dataset = datasets.ImageFolder(f"{DATA_DIR}/test", transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"Tester: {CHECKPOINT}")
print(f"Antal testbilleder: {len(test_dataset)}")

# Byg samme arkitektur, og fyld de trænede vægte ind
model = make_model(MODEL_NAME, pretrained=False)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

# Saml alle rigtige labels, gæt og sandsynligheder for hele testsættet
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images).squeeze(1)              # fra form [32, 1] til [32]
        probabilities = torch.sigmoid(outputs)          # score -> sandsynlighed for rotten (0-1)
        predictions = (probabilities > 0.5).long()      # over 50 % = rotten (1), ellers fresh (0)

        all_labels.extend(labels.tolist())
        all_preds.extend(predictions.cpu().tolist())

# Beregn metrikker (rotten = 1 er den positive klasse)
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")

# Confusion matrix, gemt som billede til rapporten
os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(3.2, 3.2))
ConfusionMatrixDisplay.from_predictions(
    all_labels, all_preds,
    display_labels=["fresh", "rotten"],
    cmap="Blues", colorbar=False, ax=ax,
)
plt.savefig(f"figures/{run_name}_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()


# Gradcam delen:
# Grad-CAM: kig på det sidste konvolutionslag i ResNet
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# Vælg op til 4 forkerte og 4 tilfældige rigtige billeder
wrong = [i for i in range(len(all_labels)) if all_labels[i] != all_preds[i]]
correct = [i for i in range(len(all_labels)) if all_labels[i] == all_preds[i]]
random.seed(0)
chosen = wrong[:4] + random.sample(correct, 4)

names = ["fresh", "rotten"]
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

for i in chosen:
    image, label = test_dataset[i]
    target = [BinaryClassifierOutputTarget(all_preds[i])]
    heatmap = cam(input_tensor=image.unsqueeze(0).to(device), targets=target)[0]
    rgb = np.clip(image.permute(1, 2, 0).numpy() * std + mean, 0, 1).astype(np.float32)
    overlay = show_cam_on_image(rgb, heatmap, use_rgb=True)
    original = (rgb * 255).astype(np.uint8)          # originalbilledet, samme format som overlay
    combined = np.hstack([original, overlay])         # original til venstre, heatmap til højre
    plt.imsave(f"figures/gradcam_{run_name}_{i}_true-{names[label]}_pred-{names[all_preds[i]]}.png", combined)