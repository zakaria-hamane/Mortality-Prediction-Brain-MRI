## README: Image Extraction and Classification Project

### Overview

This project involves processing MRI image data, extracting relevant features using VGG16 models, and classifying the data using XGBoost models. The data used in this project is sourced from the [UCSF-PDGM collection](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/).

### Directory Structure
- `rawdata/`: Download MRI archives.
- `embeddings/`: Contains embeddings extracted from MRI images.
- `models/`: Contains trained models for image classification.
  - `image_classifier/`: VGG16 models for different MRI types.
- `metrics/`: Contains metrics for model performance.
- `tabular_data/`: Metadata associated with the MRI images.
- `images_preprocessing.py`: Script to preprocess images.
- `extract_images.py`: Script to extract images from `.nii.gz` files.
- `extract_embedings_vgg16.py`: Script to extract embeddings using VGG16.
- `train_vgg16_classifier.py`: Script to train VGG16 classifier.
- `train_xgboost_classifier.py`: Script to train XGBoost classifier.
- `train_xgboost_classifier_onlyembeddings.py`: Script to train XGBoost classifier using only embeddings.
- `test_vgg16.py`: Script to test VGG16 classifier.
- `shap_xgboost.py`: Script for SHAP analysis on XGBoost model.
- `requirements.txt`: Dependencies required for the project.

### Step-by-Step Guide

#### 1. Clone the Repository

Clone the repository to your local machine.

```bash
git clone https://github.com/zakaria-hamane/Mortality-Prediction-Brain-MRI
cd Mortality-Prediction-Brain-MRI
```

#### 2. Set Up the Virtual Environment

Set up and activate a virtual environment.

```bash
python3 -m venv myenv
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
```

#### 3. Install Dependencies

Install the required Python packages.

```bash
pip install -r requirements.txt
```

#### 4. Organize and Extract Images

Run the script to extract images from `.nii.gz` files and organize them into directories.

```bash
python extract_images.py
```

#### 5. Preprocess Images

Run the preprocessing script to convert `.nii.gz` files to `.png` images.

```bash
python images_preprocessing.py
```

#### 6. Extract Embeddings

Use the VGG16 models to extract embeddings from the images.

```bash
python extract_embedings_vgg16.py
```

#### 7. Train VGG16 Classifier

Train the VGG16 classifier on the extracted images.

```bash
python train_vgg16_classifier.py
```

![image](https://github.com/user-attachments/assets/5834b058-73bc-460d-b0f7-b0f9ca81d34d)

#### 8. Train XGBoost Classifier

Train the XGBoost classifier on the extracted embeddings.

```bash
python train_xgboost_classifier.py
```

![image](https://github.com/user-attachments/assets/33ebc5d5-5b5f-40e2-8ecc-5d4ffaed0408)

#### 9. Train XGBoost Classifier Using Only Embeddings

Train the XGBoost classifier using only embeddings.

```bash
python train_xgboost_classifier_onlyembeddings.py
```

#### 10. Test VGG16 Classifier

Test the VGG16 classifier on the validation set.

```bash
python test_vgg16.py
```

#### 11. SHAP Analysis on XGBoost Model

Perform SHAP analysis to interpret the XGBoost model.

```bash
python shap_xgboost.py
```

Trained Models
You can download the pre-trained models from the following link: [Trained Models](https://drive.google.com/file/d/1fr77LGaIuRwkGLpc9RX4pfZYGcsN2Mhl/view?usp=drive_link).

### Notes

- Ensure that the data directory structure matches the expected format.
- The models and metrics directories will be populated after running the training scripts.
- Modify the scripts as needed to suit your specific requirements.

### References

- UCSF-PDGM collection: [Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/)

### Contact

For any issues or questions, please contact [zakaria.hamane1@gmail.com](mailto:zakaria.hamane1@gmail.com).

---

This README provides a comprehensive guide to set up and run the project, ensuring all steps and file locations are clearly explained.
