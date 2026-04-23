"""
Skrypt treningowy YOLOv8 do detekcji elementów rzutu architektonicznego.
Trenuje model na lokalnym datasecie wyeksportowanym z Roboflow w formacie YOLOv8.

Użycie:
    1. Wyeksportuj dataset z Roboflow w formacie YOLOv8 do folderu yolo_dataset/
    2. Uruchom: python train_yolo.py
    3. Skopiuj najlepszy model: cp runs/detect/yolo_floorplan/weights/best.pt yolo_floorplan.pt
"""

from ultralytics import YOLO


def train_yolo():
    # YOLOv8n (nano) — najlżejszy wariant, odpowiedni dla ~84 obrazów treningowych
    model = YOLO("yolov8n.pt")

    model.train(
        data="yolo_dataset/data.yaml",
        epochs=100,
        patience=20,          # early stopping — zatrzymuje trening jeśli brak poprawy przez 20 epok
        imgsz=640,
        batch=8,
        name="yolo_floorplan",
        verbose=True,
    )

    print("\n--- Trening zakończony ---")
    print("Najlepszy model: runs/detect/yolo_floorplan/weights/best.pt")
    print("Skopiuj go do katalogu projektu:")
    print("  cp runs/detect/yolo_floorplan/weights/best.pt yolo_floorplan.pt")


if __name__ == "__main__":
    train_yolo()
