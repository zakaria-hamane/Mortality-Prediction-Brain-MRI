import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import psutil
import time
import gc

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


# Function to load images and labels
def load_images_and_labels(mri_type, extracted_images_path):
    data = []
    labels = []

    print(f"Loading images for MRI type: {mri_type}")

    start_time = time.time()
    for patient_folder in sorted(os.listdir(extracted_images_path)):
        patient_path = os.path.join(extracted_images_path, patient_folder)
        if os.path.isdir(patient_path):
            patient_id = patient_folder.split('_')[1]
            print(f"Processing patient ID: {patient_id}")
            survival_status = metadata[metadata['ID'] == patient_id]['1-dead 0-alive'].values
            if len(survival_status) == 0:
                print(f"Survival status not found for patient ID: {patient_id}")
                continue
            label = survival_status[0]

            mri_type_folder = os.path.join(patient_path, mri_type)
            if os.path.exists(mri_type_folder):
                print(f"Loading images from {mri_type_folder}")
                patient_images = load_images_for_patient(mri_type_folder)
                if patient_images is not None:
                    data.append(patient_images)
                    labels.append(label)
                    print(f"Loaded images for patient ID: {patient_id}")
            else:
                print(f"Folder not found: {mri_type_folder}")
    end_time = time.time()
    print(f"Finished loading images. Time taken: {end_time - start_time} seconds")

    return np.array(data), np.array(labels)


# Function to build and compile the model
def build_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    base_model.trainable = False  # Freeze the base model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    mri_type = 'segmentation'  # Change this to the desired MRI type
    extracted_images_path = 'extracted_images'

    # Load all tabular_data and labels
    data, labels = load_images_and_labels(mri_type, extracted_images_path)

    # Split the tabular_data
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Build and compile the model
    model = build_model()

    # Set the model output path
    model_output_path = f'models/image_classifier/{mri_type}_vgg16'
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    # Set the checkpoint
    checkpoint = ModelCheckpoint(
        os.path.join(model_output_path, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # Train the model
    print(f"Starting model training with {len(X_train)} training samples and {len(X_val)} validation samples")
    start_train_time = time.time()
    model.fit(
        X_train, y_train,
        epochs=10,  # Increase epochs as needed
        batch_size=8,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint]
    )
    end_train_time = time.time()
    print(f"Finished model training. Time taken: {end_train_time - start_train_time} seconds")

    # Save the final model
    model.save(os.path.join(model_output_path, 'final_model.h5'))
    print(f'Final model saved at {model_output_path}')
