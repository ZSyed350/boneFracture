# Bone Fracture Detection Project

## Current Status

### EDA & Data Cleaning (`EDA.ipynb`)
- Added package installation cell for dependencies
- Downloaded bone fracture dataset from Kaggle
- Analyzed class distribution across train/val/test splits
- **Data cleaning performed:**
  - Merged "humerus" class (ID 4) → "humerus fracture" (ID 3)
  - Reindexed remaining classes (5→4, 6→5)
  - Removed 150 "fingers positive" samples (class imbalance)
  - Augmented 75 "wrist positive" samples (data augmentation)
- Updated `data.yaml` to reflect 6 classes (removed "humerus")

### Model Training (`train_model.ipynb`)
- Trained YOLOv8n baseline model on cleaned dataset
- **Results:**
  - mAP50: 0.318 (31.8%)
  - mAP50-95: 0.119 (11.9%)
  - Precision: 0.410, Recall: 0.348
  - **Best classes:** Forearm fracture (0.547), Shoulder fracture (0.552)
  - **Worst class:** Elbow positive (0.034)
- Training stopped early at epoch 71 (patience=20), best model at epoch 51
- Model saved to `runs/detect/bone_fracture_yolov8/weights/best.pt`

## Future Improvements
1. **Larger model:** Try YOLOv8s or YOLOv8m for better accuracy
2. **More training:** Increase epochs or disable early stopping
3. **Address class imbalance:** Elbow positive needs more data/augmentation
4. **Hyperparameter tuning:** Adjust learning rate, batch size, augmentation
5. **Data quality:** Verify labels, especially for underperforming classes

## Files
- `EDA.ipynb` - Exploratory data analysis and cleaning
- `train_model.ipynb` - YOLOv8 model training and evaluation
- `BoneFractureYolo8/data.yaml` - Dataset configuration (6 classes)
- `runs/` - Training outputs (excluded from git, see `.gitignore`)
