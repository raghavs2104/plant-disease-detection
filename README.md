cd# Plant Disease Detection using Transfer Learning

A complete PyCharm project that classifies plant-leaf images using
MobileNetV2 transfer learning.

## Features

- Automatic detection of common PlantVillage folder layouts
- Stratified 70% training, 15% validation and 15% test split
- MobileNetV2 pretrained on ImageNet
- Frozen transfer-learning stage followed by optional fine-tuning
- Data augmentation and class weighting
- Automatic selection of the better validation model
- Test accuracy, classification report and confusion matrix
- Single-image prediction script
- Tkinter desktop application

## Important limitation

This model only predicts classes included in the training dataset. It is an
educational classifier, not a laboratory diagnosis system. Clean PlantVillage
images also differ from real farm photographs, so strong test accuracy can
still produce weak real-world results.

## 1. Open in PyCharm

1. Extract the ZIP.
2. Open the `plant_disease_transfer_learning` folder in PyCharm.
3. Use a 64-bit Python 3.10, 3.11, 3.12 or 3.13 interpreter.
4. Create a virtual environment.

## 2. Install dependencies

Open the PyCharm terminal in the project folder:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify TensorFlow:

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

## 3. Add the dataset

Recommended PlantVillage structure:

```text
plant_disease_transfer_learning/
└── dataset/
    └── PlantVillage/
        └── color/
            ├── Apple___Apple_scab/
            ├── Apple___healthy/
            ├── Potato___Early_blight/
            ├── Potato___healthy/
            ├── Tomato___Late_blight/
            └── Tomato___healthy/
```

Each class folder must contain images. The loader also accepts class folders
directly under `dataset`, or under `dataset/train`.

For a manageable college project, begin with one crop and roughly 4-8 classes
rather than all PlantVillage classes.

## 4. Check the dataset

```powershell
python check_dataset.py
```

Custom location:

```powershell
python check_dataset.py --dataset "R:\PlantVillage"
```

## 5. Train

```powershell
python train.py
```

Custom location:

```powershell
python train.py --dataset "R:\PlantVillage"
```

Lighter CPU run:

```powershell
python train.py --batch-size 16 --transfer-epochs 8 --fine-tune-epochs 4
```

Skip fine-tuning:

```powershell
python train.py --fine-tune-epochs 0
```

The first default run downloads MobileNetV2 ImageNet weights. If that download
fails, fix the internet connection and retry. This fallback trains without
pretrained weights, but it is not proper pretrained transfer learning:

```powershell
python train.py --weights none
```

## 6. Run the desktop application

After training:

```powershell
python app.py
```

## 7. Predict from the terminal

```powershell
python predict.py --image "test_images\leaf.jpg"
```

Without an image argument, a file-selection window opens:

```powershell
python predict.py
```

## 8. Evaluate again

```powershell
python evaluate.py
```

## Generated files

```text
models/
├── plant_disease_model.keras
├── transfer_stage_model.keras
├── class_names.json
└── model_info.json

outputs/
├── dataset_split.csv
├── transfer_training_log.csv
├── fine_tune_training_log.csv
├── combined_history.csv
├── training_accuracy.png
├── training_loss.png
├── classification_report.txt
├── confusion_matrix.png
└── test_predictions.csv
```

## Common errors

### Could not locate plant-disease class folders

The dataset is missing, still zipped or nested incorrectly. Find the folder
whose immediate child folders are disease classes, then pass that folder:

```powershell
python train.py --dataset "FULL\PATH\TO\CLASS_FOLDER_PARENT"
```

### Could not open requirements file

Your terminal is in the wrong directory:

```powershell
cd "R:\plant_disease_transfer_learning"
dir
pip install -r requirements.txt
```

### No TensorFlow GPU detected

Current TensorFlow versions do not provide CUDA GPU acceleration on native
Windows. Training still works on the CPU. WSL2 is required for current
official TensorFlow NVIDIA GPU support.

### Out-of-memory error

Reduce the batch size:

```powershell
python train.py --batch-size 8
```
