import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import multiprocessing as mp
import json

# Enable memory growth for the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load metadata
metadata_path = 'tabular_data/UCSF-PDGM-metadata_v2.csv'
metadata = pd.read_csv(metadata_path)
metadata['ID'] = metadata['ID'].astype(str)
metadata['ID'] = metadata['ID'].str.extract(r'(\d+)$')
metadata['ID'] = metadata['ID'].astype(int).astype(str).str.zfill(4)

# Function to load images for a patient
def load_images_for_patient(mri_type_folder):
    images = []
    for file_name in sorted(os.listdir(mri_type_folder)):
        if file_name.endswith('.png'):
            image_path = os.path.join(mri_type_folder, file_name)
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))  # Resize to match VGG16 input size
            img_array = np.array(img)
            img_array = preprocess_input(img_array)
            images.append(img_array)
    if images:
        # Average images along the depth dimension
        return np.mean(np.stack(images, axis=0), axis=0)
    return None

# Function to process a single patient's tabular_data
def process_patient_data(args):
    patient_folder, extracted_images_path, mri_type = args
    patient_path = os.path.join(extracted_images_path, patient_folder)
    if os.path.isdir(patient_path):
        patient_id = patient_folder.split('_')[1]
        survival_status = metadata[metadata['ID'] == patient_id]['1-dead 0-alive'].values
        if len(survival_status) == 0:
            return None, None
        label = survival_status[0]

        mri_type_folder = os.path.join(patient_path, mri_type)
        if not os.path.exists(mri_type_folder):
            return None, None

        patient_images = load_images_for_patient(mri_type_folder)

        if patient_images is None:
            return None, None

        # Reshape the patient images to match model input
        patient_images = np.expand_dims(patient_images, axis=0)

        return patient_images, label
    return None, None

# Function to evaluate the model on a list of MRI types
def evaluate_model_on_mri_types(mri_types, extracted_images_path):
    results = {}

    for mri_type in mri_types:
        model_path = f'models/image_classifier/{mri_type}_vgg16/final_model.h5'

        if not os.path.exists(model_path):
            print(f"Model for {mri_type} not found at {model_path}")
            continue

        # Load model
        model = load_model(model_path)
        print(f"Loaded model from {model_path}")

        all_labels = []
        all_predictions = []

        # Create a pool of workers to load and process images in parallel
        pool = mp.Pool(mp.cpu_count())
        patient_folders = sorted(os.listdir(extracted_images_path))
        args = [(patient_folder, extracted_images_path, mri_type) for patient_folder in patient_folders]

        # Process patients in parallel
        results_list = pool.map(process_patient_data, args)

        for patient_images, label in results_list:
            if patient_images is not None and label is not None:
                # Make prediction
                prediction = model.predict(patient_images)
                predicted_label = 1 if prediction[0][0] > 0.5 else 0

                all_labels.append(label)
                all_predictions.append(predicted_label)

        pool.close()
        pool.join()

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)

        results[mri_type] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        print(f"Results for {mri_type}:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

    # Save results to a JSON file
    with open('metrics/vgg16_metrics.json', 'w') as f:
        json.dump(results, f)
    print("Results saved to vgg16_metrics.json")

if __name__ == "__main__":
    mri_types = [
        'bias',
        'eddy_FA',
        'eddy_L1',
        'eddy_L2',
        'eddy_L3',
        'eddy_MD',
        'misc',
        'parenchyma_segmentation',
        'segmentation'
    ]
    extracted_images_path = 'extracted_images'

    evaluate_model_on_mri_types(mri_types, extracted_images_path)
