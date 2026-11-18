
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# Paths
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "mindease_emotion_model_v3.keras"
)

RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = [
    "angry",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# Load test dataset
# Explicitly select the five classes used by V3
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASS_NAMES,
    shuffle=False
)

class_names = test_dataset.class_names

print("\nClasses:", class_names)

# Load V3 model
print("\nLoading MindEase CNN V3...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("V3 model loaded successfully!")
print("Model output shape:", model.output_shape)

# Predict
print("\nGenerating predictions...")

y_probabilities = model.predict(test_dataset)

y_pred = np.argmax(y_probabilities, axis=1)

# Actual labels
y_true = np.concatenate([
    np.argmax(labels.numpy(), axis=1)
    for _, labels in test_dataset
])

# Accuracy
accuracy = accuracy_score(y_true, y_pred)

print(
    "\nEvaluation Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

# Classification report
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print("\nClassification Report:\n")
print(report)

# Save classification report
report_path = os.path.join(
    RESULTS_DIR,
    "classification_report_v3.txt"
)

with open(report_path, "w") as file:
    file.write(report)

# Confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(len(class_names))
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Expression")
plt.ylabel("Actual Expression")
plt.title("MindEase CNN V3 - Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix_v3.png"
)

plt.savefig(cm_path, dpi=300)
plt.show()

print("\nV3 Evaluation completed!")
print("Classification report:", report_path)
print("Confusion matrix:", cm_path)
