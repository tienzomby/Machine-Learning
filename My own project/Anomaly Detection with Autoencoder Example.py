import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Generate sample data with anomalies
np.random.seed(42)

# Create normal data following certain patterns
normal_data = np.random.normal(0, 0.5, size=(1000, 10))
# Add correlations between some features
normal_data[:, 1] = normal_data[:, 0] * 0.7 + np.random.normal(0, 0.3, 1000)
normal_data[:, 3] = normal_data[:, 2] * 0.5 + normal_data[:, 4] * 0.5 + np.random.normal(0, 0.2, 1000)

# Create anomalies - data that breaks the established patterns
anomaly_data = np.random.normal(1, 1, size=(100, 10))

# Combine and label the data
X_full = np.vstack([normal_data, anomaly_data])
y_full = np.hstack([np.zeros(1000), np.ones(100)])  # 0 for normal, 1 for anomalies

# Split into training (only normal data) and test sets
X_normal = X_full[y_full == 0]
X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)

# 2. Preprocess the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_full_scaled = scaler.transform(X_full)

# 3. Build Autoencoder model
def build_autoencoder(input_dim, encoding_dim=5):
    # Input layer
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoder = Dense(8, activation='relu')(input_layer)
    encoder = Dropout(0.2)(encoder)
    encoder = Dense(encoding_dim, activation='relu')(encoder)
    
    # Decoder
    decoder = Dense(8, activation='relu')(encoder)
    decoder = Dropout(0.2)(decoder)
    decoder = Dense(input_dim, activation='linear')(decoder)
    
    # Autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    
    # Compile model
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

# Build and train the model
input_dim = X_train_scaled.shape[1]
autoencoder = build_autoencoder(input_dim)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model
history = autoencoder.fit(
    X_train_scaled, X_train_scaled,  # Autoencoders learn to reconstruct the input
    epochs=100,
    batch_size=32,
    validation_data=(X_val_scaled, X_val_scaled),
    callbacks=[early_stopping],
    verbose=0
)

# 4. Anomaly detection process
# Predict the reconstructions
reconstructions = autoencoder.predict(X_full_scaled)

# Calculate reconstruction error (MSE) for each sample
mse = np.mean(np.square(X_full_scaled - reconstructions), axis=1)

# 5. Determine threshold for anomaly classification
# We use validation data to establish the threshold
val_reconstructions = autoencoder.predict(X_val_scaled)
val_mse = np.mean(np.square(X_val_scaled - val_reconstructions), axis=1)

# Set threshold as mean + 2*std of validation errors
threshold = np.mean(val_mse) + 2 * np.std(val_mse)
print(f"Anomaly threshold: {threshold:.6f}")

# 6. Classify anomalies based on threshold
predicted_anomalies = mse > threshold

# 7. Evaluate the model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

print("\nConfusion Matrix:")
print(confusion_matrix(y_full, predicted_anomalies))
print("\nClassification Report:")
print(classification_report(y_full, predicted_anomalies))

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_full, mse)
roc_auc = auc(fpr, tpr)

# 8. Visualize results
plt.figure(figsize=(14, 5))

# Plot reconstruction error
plt.subplot(1, 2, 1)
plt.hist(mse[y_full == 0], bins=50, alpha=0.5, label='Normal')
plt.hist(mse[y_full == 1], bins=50, alpha=0.5, label='Anomaly')
plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
plt.xlabel('Reconstruction Error (MSE)')
plt.ylabel('Count')
plt.legend()
plt.title('Reconstruction Error Distribution')

# Plot ROC curve
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

plt.tight_layout()
plt.show()

# 9. Function to detect anomalies in new data
def detect_anomalies(new_data, autoencoder, scaler, threshold):
    # Preprocess
    scaled_data = scaler.transform(new_data)
    
    # Get reconstructions
    reconstructions = autoencoder.predict(scaled_data)
    
    # Calculate errors
    mse = np.mean(np.square(scaled_data - reconstructions), axis=1)
    
    # Classify
    anomalies = mse > threshold
    
    return anomalies, mse

# Example usage with new data
new_data = np.random.normal(0, 0.5, size=(5, 10))  # Normal data
new_anomalies = np.random.normal(2, 1, size=(5, 10))  # Anomalous data
combined_new = np.vstack([new_data, new_anomalies])

# Detect anomalies
anomaly_predictions, error_scores = detect_anomalies(combined_new, autoencoder, scaler, threshold)

print("\nPredictions for new data:")
for i, (pred, score) in enumerate(zip(anomaly_predictions, error_scores)):
    status = "ANOMALY" if pred else "normal"
    print(f"Sample {i+1}: {status} (Error score: {score:.6f}, Threshold: {threshold:.6f})")
