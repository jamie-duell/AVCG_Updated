from main import get_dataset
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datasets = ["breast_cancer", "heart_disease", "spambase", "adult_income"]
for ds_name in datasets:
    try:
        train, val, test, scaler, feats, classes, lower, upper = get_dataset(ds_name)
        print(f"Dataset {ds_name} loaded successfully.")
        print(f"  Features: {len(feats)}")
        print(f"  Classes: {classes}")
        print(f"  Train size: {len(train)}")
    except Exception as e:
        print(f"Error loading {ds_name}: {e}")
