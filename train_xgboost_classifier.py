import os
import numpy as np
import pandas as pd
import json
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pickle

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

# Split tabular_data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost classifier
model = xgb.XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Save the model
model_filename = 'models/xgboost_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {model_filename}")

# Save the metrics to a JSON file
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1
}

metrics_filename = 'metrics/xgboost_metrics.json'
with open(metrics_filename, 'w') as f:
    json.dump(metrics, f)
print(f"Metrics saved to {metrics_filename}")
