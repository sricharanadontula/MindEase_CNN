
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.utils.class_weight import compute_class_weight

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(MODEL_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 8
NUM_CLASSES = 7
SEED = 42

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "mindease_emotion_model_v2.keras"
)

# --------------------------------------------------
# LOAD TRAINING AND VALIDATION DATASETS
# --------------------------------------------------

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = train_dataset.class_names

print("\nEmotion classes:", class_names)

if len(class_names) != NUM_CLASSES:
    raise ValueError("Expected exactly 7 emotion classes.")

if class_names != val_dataset.class_names:
    raise ValueError("Training and validation classes do not match.")

# --------------------------------------------------
# CALCULATE CLASS WEIGHTS
# --------------------------------------------------

print("\nCalculating class weights...")

train_labels = np.concatenate([
    np.argmax(labels.numpy(), axis=1)
    for _, labels in train_dataset
])

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_labels
)

print("\nClass weights:")

for name, weight in zip(class_names, class_weights):
    print(f"{name}: {weight:.3f}")

# Apply weights to each training example.
class_weights_tensor = tf.constant(
    class_weights,
    dtype=tf.float32
)

def add_sample_weights(images, labels):
    label_indices = tf.argmax(labels, axis=1)

    sample_weights = tf.gather(
        class_weights_tensor,
        label_indices
    )

    return images, labels, sample_weights

train_dataset = train_dataset.map(
    add_sample_weights,
    num_parallel_calls=tf.data.AUTOTUNE
)

# --------------------------------------------------
# OPTIMIZE DATASETS
# --------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
val_dataset = val_dataset.prefetch(AUTOTUNE)

# --------------------------------------------------
# DATA AUGMENTATION
# --------------------------------------------------

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1)
])

# --------------------------------------------------
# LOAD MOBILENETV2
# --------------------------------------------------

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# --------------------------------------------------
# BUILD CNN MODEL
# --------------------------------------------------

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# Equivalent to MobileNetV2 preprocess_input
# for images with pixel values from 0 to 255.
x = layers.Rescaling(
    1.0 / 127.5,
    offset=-1
)(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# --------------------------------------------------
# INITIAL TRAINING
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )
]

print("\nStarting initial training...")

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks
)

# --------------------------------------------------
# FINE-TUNING MOBILENETV2
# --------------------------------------------------

print("\nStarting MobileNetV2 fine-tuning...")

base_model.trainable = True

# Freeze earlier layers and fine-tune the last 30.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep Batch Normalization layers frozen.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
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

model.fit(
    train_dataset,
    validation_data=val_dataset,
    initial_epoch=INITIAL_EPOCHS,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    callbacks=fine_tune_callbacks
)

# --------------------------------------------------
# FINAL TEST EVALUATION
# --------------------------------------------------

print("\nLoading best saved V2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("\nLoading untouched test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_loss, test_accuracy = model.evaluate(test_dataset)

print("\nMindEase CNN V2 Training Completed!")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\nModel saved at:")
print(MODEL_PATH)
