import os
import gzip
import shutil
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


# Function to extract .nii.gz files
def extract_nii_gz(file_path, dest_path):
    with gzip.open(file_path, 'rb') as f_in:
        with open(dest_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


# Function to convert .nii files to .png images
def nii_to_png(nii_file, output_folder):
    img = nib.load(nii_file)
    data = img.get_fdata()

    # Normalize the tabular_data to [0, 255] for PNG
    data = 255 * (data - np.min(data)) / (np.max(data) - np.min(data))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save each slice as a PNG image
    for i in range(data.shape[2]):
        slice_data = data[:, :, i]
        # Ensure slice_data is 2D and float
        if slice_data.ndim == 2:
            plt.imsave(os.path.join(output_folder, f'slice_{i:03d}.png'), slice_data.astype(np.uint8), cmap='gray')


# Function to get the last processed patient number
def get_last_processed_patient(extracted_images_path):
    if not os.path.exists(extracted_images_path):
        return 0

    patient_numbers = []
    for folder_name in os.listdir(extracted_images_path):
        if folder_name.startswith('patient_'):
            try:
                patient_number = int(folder_name.split('_')[1])
                patient_numbers.append(patient_number)
            except (IndexError, ValueError):
                continue

    return max(patient_numbers, default=0)


# Function to extract patient number from folder name
def extract_patient_number(folder_name):
    try:
        patient_number = int(folder_name.split('-')[2].split('_')[0])
        return patient_number
    except (IndexError, ValueError):
        print(f"Unexpected folder name format: {folder_name}")
        return None


# Main function to organize, extract and convert files
def organize_and_extract_images(rawdata_path):
    extracted_images_path = 'extracted_images'
    last_processed_patient = get_last_processed_patient(extracted_images_path)

    for patient_folder in sorted(os.listdir(rawdata_path)):
        patient_path = os.path.join(rawdata_path, patient_folder)

        if os.path.isdir(patient_path):
            patient_number = extract_patient_number(patient_folder)
            if patient_number is None or patient_number <= last_processed_patient:
                continue

            extracted_patient_folder = os.path.join(extracted_images_path, f'patient_0{patient_number}_nifti')

            if not os.path.exists(extracted_patient_folder):
                os.makedirs(extracted_patient_folder)

            for file_name in os.listdir(patient_path):
                if file_name.endswith('.nii.gz'):
                    # Handle cases where the format might be different
                    file_parts = file_name.split('_')
                    if len(file_parts) < 3:
                        print(f"Unexpected file name format: {file_name}")
                        image_type = 'misc'
                    else:
                        image_type = '_'.join(file_parts[2:]).replace('.nii.gz', '')

                    extracted_image_folder = os.path.join(extracted_patient_folder, image_type)

                    if not os.path.exists(extracted_image_folder):
                        os.makedirs(extracted_image_folder)

                    source_file = os.path.join(patient_path, file_name)
                    extracted_file = os.path.join(extracted_image_folder, file_name.replace('.gz', ''))

                    # Extract and convert to PNG
                    extract_nii_gz(source_file, extracted_file)
                    nii_to_png(extracted_file, extracted_image_folder)
                    os.remove(extracted_file)  # Remove the .nii file after conversion
                    print(f'Extracted and converted: {extracted_file}')


if __name__ == "__main__":
    rawdata_path = 'rawdata'
    organize_and_extract_images(rawdata_path)
