import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define the raw data directory
raw_data_dir = 'rawdata'


# Function to extract and save slices from nii.gz file
def save_slices(nii_file, output_folder):
    img = nib.load(nii_file)
    data = img.get_fdata()

    # Normalize data to 0-1 range
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Check if the data is 3D or 4D
    if data.ndim == 3:
        for i in range(data.shape[2]):
            slice_data = data[:, :, i]
            plt.imshow(slice_data.T, cmap='gray', origin='lower')
            plt.axis('off')
            plt.savefig(os.path.join(output_folder, f'slice_{i:03d}.png'), bbox_inches='tight', pad_inches=0)
            plt.close()
    elif data.ndim == 4:
        for i in range(data.shape[3]):
            for j in range(data.shape[2]):
                slice_data = data[:, :, j, i]
                plt.imshow(slice_data.T, cmap='gray', origin='lower')
                plt.axis('off')
                plt.savefig(os.path.join(output_folder, f'slice_{i:03d}_{j:03d}.png'), bbox_inches='tight',
                            pad_inches=0)
                plt.close()

    print(f"Finished processing {nii_file}")


# Function to process each file
def process_file(patient_id, file_name):
    patient_dir = os.path.join(raw_data_dir, patient_id)
    nii_file = os.path.join(patient_dir, file_name)
    output_patient_dir = os.path.join('images', patient_id)
    os.makedirs(output_patient_dir, exist_ok=True)
    folder_name = os.path.splitext(os.path.splitext(file_name)[0])[0]  # Remove .nii.gz
    output_folder = os.path.join(output_patient_dir, folder_name)

    # Create folder for the file
    os.makedirs(output_folder, exist_ok=True)

    # Extract and save slices
    save_slices(nii_file, output_folder)


# Main execution
if __name__ == '__main__':
    with ThreadPoolExecutor() as executor:
        futures = []
        for patient_id in os.listdir(raw_data_dir):
            patient_dir = os.path.join(raw_data_dir, patient_id)
            if os.path.isdir(patient_dir):
                for file_name in os.listdir(patient_dir):
                    if file_name.endswith('.nii.gz'):
                        futures.append(executor.submit(process_file, patient_id, file_name))

        for future in as_completed(futures):
            future.result()

    print("Processing complete.")
