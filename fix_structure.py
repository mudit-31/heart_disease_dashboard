import os
import shutil

# Define correct paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Ensure 'models' and 'data' directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Fix heart.csv location
old_data_path = os.path.join(MODELS_DIR, "data", "heart.csv")
new_data_path = os.path.join(DATA_DIR, "heart.csv")

if os.path.exists(old_data_path):
    shutil.move(old_data_path, new_data_path)
    print(f"Moved dataset to: {new_data_path}")

# Fix model.pkl location
old_model_path = os.path.join(MODELS_DIR, "models", "heart_disease_model.pkl")
new_model_path = os.path.join(MODELS_DIR, "heart_disease_model.pkl")

if os.path.exists(old_model_path):
    shutil.move(old_model_path, new_model_path)
    print(f"Moved model to: {new_model_path}")

# Clean up empty directories
try:
    os.rmdir(os.path.join(MODELS_DIR, "data"))  # Remove empty 'models/data'
    os.rmdir(os.path.join(MODELS_DIR, "models"))  # Remove empty 'models/models'
except OSError:
    pass  # Skip if not empty

print("✅ File structure corrected successfully!")
