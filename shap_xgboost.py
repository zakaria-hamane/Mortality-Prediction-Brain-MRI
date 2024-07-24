import os
import numpy as np
import pandas as pd
import json
import shap
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt

# Load metadata
metadata_path = 'tabular_data/UCSF-PDGM-metadata_v2.csv'
metadata = pd.read_csv(metadata_path)
metadata['ID'] = metadata['ID'].astype(str)
metadata['ID'] = metadata['ID'].str.extract(r'(\d+)$')
metadata['ID'] = metadata['ID'].astype(int).astype(str).str.zfill(4)

# Define MRI types
mri_types = [
    "bias", "eddy_FA", "eddy_L1", "eddy_L2", "eddy_L3", "eddy_MD",
    "misc", "parenchyma_segmentation", "segmentation"
]

# Define additional attributes to be used from metadata
additional_attributes = ['Sex', 'Age at MRI']

# Ensure 'Sex' is encoded as numeric
metadata['Sex'] = metadata['Sex'].map({'M': 0, 'F': 1})

# Load embeddings and corresponding labels
X = []
y = []

for mri_type in mri_types:
    embeddings_file = f'embeddings/embeddings_{mri_type}.json'
    if not os.path.exists(embeddings_file):
        print(f"Embeddings file not found for MRI type: {mri_type}")
        continue

    with open(embeddings_file, 'r') as f:
        embeddings_data = json.load(f)

    for entry in embeddings_data:
        patient_id = entry['patient_id']
        embeddings = entry['embeddings']

        if len(embeddings) == 0:
            print(f"No embeddings found for patient ID: {patient_id}")
            continue

        if metadata[metadata['ID'] == patient_id].empty:
            print(f"Metadata not found for patient ID: {patient_id}")
            continue

        patient_metadata = metadata[metadata['ID'] == patient_id][additional_attributes].values.flatten().tolist()
        if len(patient_metadata) != len(additional_attributes):
            print(f"Incomplete metadata for patient ID: {patient_id}")
            continue

        combined_features = np.concatenate([embeddings[0], patient_metadata])
        X.append(combined_features)

        survival_status = metadata[metadata['ID'] == patient_id]['1-dead 0-alive'].values[0]
        y.append(survival_status)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Load the trained XGBoost model
model_filename = 'models/xgboost_model.pkl'
with open(model_filename, 'rb') as f:
    model = pickle.load(f)

# Initialize the SHAP explainer
explainer = shap.Explainer(model, X)

# Calculate SHAP values
shap_values = explainer(X)

# Save SHAP values for further inspection
shap_values_filename = 'shap_values.npy'
np.save(shap_values_filename, shap_values.values)
print(f"SHAP values saved to {shap_values_filename}")

# Plot summary of feature importances
shap.summary_plot(shap_values, X, feature_names=[f"embedding_{i}" for i in range(
    X.shape[1] - len(additional_attributes))] + additional_attributes)
plt.savefig('shap_summary_plot.png')
print("SHAP summary plot saved to shap_summary_plot.png")

# Plot feature importance bar chart
shap.summary_plot(shap_values, X, plot_type="bar", feature_names=[f"embedding_{i}" for i in range(
    X.shape[1] - len(additional_attributes))] + additional_attributes)
plt.savefig('shap_feature_importance_bar_chart.png')
print("SHAP feature importance bar chart saved to shap_feature_importance_bar_chart.png")
