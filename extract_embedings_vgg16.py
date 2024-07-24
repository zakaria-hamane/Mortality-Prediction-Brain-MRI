import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
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


def extract_embeddings(mri_type, extracted_images_path, model_path):
    # Load model
    model = load_model(model_path)
    model = Model(inputs=model.input, outputs=model.layers[-2].output)  # Use the second last layer for embeddings
    print(f"Loaded model from {model_path}")

    results = []

    for patient_folder in sorted(os.listdir(extracted_images_path)):
        patient_path = os.path.join(extracted_images_path, patient_folder)
        if os.path.isdir(patient_path):
            patient_id = patient_folder.split('_')[1]
            mri_type_folder = os.path.join(patient_path, mri_type)
            if not os.path.exists(mri_type_folder):
                print(f"Folder not found: {mri_type_folder}")
                continue

            patient_images = load_images_for_patient(mri_type_folder)

            if patient_images is None:
                print(f"No images found for patient ID: {patient_id}")
                continue

            # Reshape the patient images to match model input
            patient_images = np.expand_dims(patient_images, axis=0)

            # Get embeddings
            embeddings = model.predict(patient_images)

            result = {
                'patient_id': patient_id,
                'embeddings': embeddings.tolist()
            }
            results.append(result)

    return results


if __name__ == "__main__":
    mri_types = [
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

    for mri_type in mri_types:
        model_path = f'models/image_classifier/{mri_type}_vgg16/final_model.h5'

        if not os.path.exists(model_path):
            print(f"Model for {mri_type} not found at {model_path}")
            continue

        results = extract_embeddings(mri_type, extracted_images_path, model_path)

        if results:
            output_path = f'embeddings/embeddings_{mri_type}.json'
            with open(output_path, 'w') as f:
                json.dump(results, f)
            print(f"Embeddings saved to {output_path}")
