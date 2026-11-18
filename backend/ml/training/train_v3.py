
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight

# ------------------------------------------------
# PATHS AND CONFIGURATION
# ------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

V2_PATH = os.path.join(
    BASE_DIR, "saved_models",
    "mindease_emotion_model_v2.keras"
)

V3_PATH = os.path.join(
    BASE_DIR, "saved_models",
    "mindease_emotion_model_v3.keras"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5
SEED = 42

# Keep these five classes in this exact order.
CLASS_NAMES = [
    "angry",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# ------------------------------------------------
# LOAD FIVE-CLASS DATASETS
# ------------------------------------------------

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

print("\nSelected classes:", CLASS_NAMES)

# ------------------------------------------------
# CLASS WEIGHTS
# ------------------------------------------------

train_labels = np.concatenate([
    np.argmax(labels.numpy(), axis=1)
    for _, labels in train_dataset
])

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_labels
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(weights)
}

print("\nClass weights:", class_weights)

# ------------------------------------------------
# LOAD V2 AND REUSE MOBILENETV2
# ------------------------------------------------

print("\nLoading V2 model...")

old_model = tf.keras.models.load_model(V2_PATH)

base_model = old_model.get_layer(
    "mobilenetv2_1.00_224"
)

base_model.trainable = False

# ------------------------------------------------
# BUILD V3 WITH A NEW FIVE-CLASS HEAD
# ------------------------------------------------

inputs = layers.Input(shape=(224, 224, 3))

# Reuse V2 augmentation and preprocessing.
x = old_model.get_layer("sequential")(inputs)
x = old_model.get_layer("rescaling")(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="emotion_output"
)(x)

model = models.Model(inputs, outputs)

# ------------------------------------------------
# FIRST TRAINING STAGE
# ------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        V3_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

print("\nStarting five-class training...")

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=8,
    class_weight=class_weights,
    callbacks=callbacks
)

# ------------------------------------------------
# FINE-TUNING
# ------------------------------------------------

print("\nStarting fine-tuning...")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.00001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

fine_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        V3_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )
]

model.fit(
    train_dataset,
    validation_data=val_dataset,
    initial_epoch=8,
    epochs=13,
    class_weight=class_weights,
    callbacks=fine_callbacks
)

# ------------------------------------------------
# FINAL TEST EVALUATION
# ------------------------------------------------

print("\nLoading best V3 model...")

best_model = tf.keras.models.load_model(V3_PATH)

test_loss, test_accuracy = best_model.evaluate(test_dataset)

print("\nMindEase CNN V3 completed!")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print("Saved model:", V3_PATH)
