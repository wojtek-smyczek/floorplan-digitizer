import os
import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model_wojtka.pth"


def get_trained_model():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 10)
    )
    model.to(device)

    if os.path.exists(MODEL_PATH):
        print(f"--- Ładowanie modelu: {MODEL_PATH} ---")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f"BŁĄD: Nie znaleziono pliku {MODEL_PATH}!")
        exit()

    model.eval()
    return model
