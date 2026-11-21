import sys, os
from download_data import downloadUAVDT

if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(ROOT, "data")
    models_path = os.path.join(ROOT, "models")

    # Download the dataset if not already in data path
    downloadUAVDT(data_path)

    # Preprocess the data

    # Yolo model training

    # RF-Detr model training
