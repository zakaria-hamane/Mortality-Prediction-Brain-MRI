import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load metadata
metadata_path = 'tabular_data/UCSF-PDGM-metadata_v2.csv'
metadata = pd.read_csv(metadata_path)
metadata['ID'] = metadata['ID'].astype(str)
metadata['ID'] = metadata['ID'].str.extract(r'(\d+)$')
metadata['ID'] = metadata['ID'].astype(int).astype(str).str.zfill(4)

# Load embeddings for all MRI types
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

embeddings = {}
embedding_length = None

for mri_type in mri_types:
    embeddings_path = f'embeddings_{mri_type}.json'
    if os.path.exists(embeddings_path):
        with open(embeddings_path, 'r') as f:
            embeddings[mri_type] = json.load(f)
            if not embedding_length and embeddings[mri_type]:
                embedding_length = len(embeddings[mri_type][0]['embeddings'][0])
    else:
        print(f"Embeddings file not found for MRI type: {mri_type}")

# Function to get the embeddings for a specific patient and MRI type
def get_patient_embeddings(patient_id, mri_type):
    if mri_type in embeddings:
        for item in embeddings[mri_type]:
            if item['patient_id'] == patient_id:
                return item['embeddings'][0]  # Assume one embedding per patient per MRI type
    return [0] * embedding_length  # Return zero vector if embedding is missing

# Combine embeddings and labels into a single dataset
X = []
y = []

for patient_id in metadata['ID']:
    patient_embeddings = []
    valid_patient = True

    for mri_type in mri_types:
        patient_emb = get_patient_embeddings(patient_id, mri_type)
        if len(patient_emb) != embedding_length:
            print(f"Skipping patient {patient_id} due to inconsistent embedding lengths in MRI type {mri_type}.")
            valid_patient = False
            break
        patient_embeddings.extend(patient_emb)

    if valid_patient:
        survival_status = metadata[metadata['ID'] == patient_id]['1-dead 0-alive'].values[0]
        X.append(patient_embeddings)
        y.append(survival_status)

X = np.array(X)
y = np.array(y)

# Check the shape of X and y
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Split the tabular_data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost classifier
model = xgb.XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Save the model
model_output_path = 'models/xgboost_model_onlyembeddings.pkl'
joblib.dump(model, model_output_path)
print(f"Model saved to {model_output_path}")

# Save the metrics to a file
metrics_output_path = 'metrics/xgboost_metrics_onlyembeddings.json'
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1
}
with open(metrics_output_path, 'w') as f:
    json.dump(metrics, f)
print(f"Metrics saved to {metrics_output_path}")
