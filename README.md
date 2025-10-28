# PointC2F Official Implementation

This is the **official implementation** of PointC2F.

## Dataset

Please download the dataset here: [Dataset Download Link](https://drive.google.com/file/d/10a3TRjzZMpEJ9Zzm7DD0vfVnZqM6jIBG/view?usp=drive_link)
You can download that and create a new folder, named `dataset/` and put dataset here.

## Environment Setup

You can set up the required environment using a `requirements.txt`.

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once the environment is activated, you can run the provided scripts for training and testing.

### Training
```bash
python train.py
```

### Testing
```bash
python test.py
```

## Pretrained Models

If you wish to evaluate the pre-trained models directly, please download them from the following links [Pretrained Link](https://drive.google.com/file/d/1JO6dYlDYPeGQErWn7vb96Qc4iiPWI2_w/view?usp=drive_link):

- **`foldX_model.pth`**: Contains the trained weights for fold X.  
  Note: Different folds use the same model architecture but are trained on different data splits.

- **`foldX_metrics.json`**: Includes evaluation results and performance metrics we obtained for the corresponding fold.