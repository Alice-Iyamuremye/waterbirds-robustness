"""Configuration shared by ERM and Group DRO training/evaluation."""
from pathlib import Path
import os
PROJECT_ROOT=Path(__file__).resolve().parent.parent
DATA_ROOT=Path(os.environ.get("WILDS_DATA_DIR","/kaggle/working/wilds_data"))
ARTIFACT_ROOT=PROJECT_ROOT/"artifacts"
MODEL_DIR=ARTIFACT_ROOT/"models"; RESULT_DIR=ARTIFACT_ROOT/"results"; SALiency_DIR=ARTIFACT_ROOT/"saliency"
for directory in (DATA_ROOT,MODEL_DIR,RESULT_DIR,SALiency_DIR): directory.mkdir(parents=True,exist_ok=True)
SEED=42; NUM_CLASSES=2; BATCH_SIZE=64; NUM_EPOCHS=50; LEARNING_RATE=1e-4; WEIGHT_DECAY=1e-4; NUM_WORKERS=2
MODEL_NAME="resnet18"; PRETRAINED=True
GROUP_NAMES=["landbird-on-land","landbird-on-water","waterbird-on-land","waterbird-on-water"]
NUM_GROUPS=len(GROUP_NAMES)
