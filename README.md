# Bone Fracture Detection Project

## Dependencies
- `kagglehub`
- `opencv-python`
- `matplotlib`
- `numpy`

## How to run the code
### EDA
1. Open Jupyter Notebook by writing `jupyter notebook` in the command line. Make sure you are already in the `ROOT` directory.
2. Run all cells.
3. Data will be available for use at the dataset path printed in cell 3.

### Training

## Folder structure

## Reproducibility instructions

## Model Training (`train_model.ipynb`)
- Trained YOLOv8n baseline model on cleaned dataset
- **Results:**
  - mAP50: 0.318 (31.8%)
  - mAP50-95: 0.119 (11.9%)
  - Precision: 0.410, Recall: 0.348
  - **Per-class metrics (validation):** AP50 = Average Precision at 50% box overlap (not the same as Precision P). Full breakdown:

    | Class              | Images | Instances | P     | R     | mAP50 | mAP50-95 |
    |--------------------|--------|-----------|-------|-------|-------|----------|
    | all                | 331    | 176       | 0.41  | 0.348 | 0.318 | 0.119    |
    | elbow positive     | 28     | 29        | 0.127 | 0.103 | 0.034 | 0.012    |
    | fingers positive   | 41     | 48        | 0.314 | 0.271 | 0.221 | 0.066    |
    | forearm fracture   | 37     | 43        | 0.687 | 0.535 | 0.558 | 0.214    |
    | shoulder fracture  | 31     | 36        | 0.669 | 0.583 | 0.557 | 0.219    |
    | wrist positive     | 19     | 20        | 0.254 | 0.250 | 0.220 | 0.084    |

    Best: forearm, shoulder. Worst: elbow.
- Training stopped early at epoch 71 (patience=20), best model at epoch 51
- Model saved to `runs/detect/bone_fracture_yolov8/weights/best.pt`

## Future Improvements
1. **Larger model:** Try YOLOv8s or YOLOv8m for better accuracy
2. **More training:** Increase epochs or disable early stopping
3. **Address class imbalance:** Elbow positive needs more data/augmentation
4. **Hyperparameter tuning:** Adjust learning rate, batch size, augmentation
5. **Data quality:** Verify labels, especially for underperforming classes