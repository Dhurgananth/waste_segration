#!/usr/bin/env python3
"""
PHASE 2 FIXED — Fine-tune properly
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = "dataset_ready"
IMG_SIZE = 300
BATCH_SIZE = 16

# ========== 1. LOAD DATA ==========
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names
print(f"Classes: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ========== 2. CLASS WEIGHTS ==========
labels = []
for _, y in train_ds:
    labels.extend(np.argmax(y.numpy(), axis=1))

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
print(f"Class weights: {class_weight_dict}")

# ========== 3. LOAD BEST PHASE 1 MODEL ==========
print("\nLoading best_model.keras from Phase 1...")
model = tf.keras.models.load_model("best_model.keras")
print("Loaded. Phase 1 peak was ~87.4%")

# ========== 4. FIND BASE MODEL BY NAME (Fixed!) ==========
# Instead of guessing layer index, find it by name
base_model = None
for layer in model.layers:
    if "efficientnet" in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    raise ValueError("Could not find EfficientNet base model in layers!")

print(f"Found base model: {base_model.name}")

# Unfreeze top 120 layers
base_model.trainable = True
for layer in base_model.layers[:-120]:
    layer.trainable = False

trainable = sum(1 for l in base_model.layers if l.trainable)
total = len(base_model.layers)
print(f"Trainable layers: {trainable} / {total}")

# ========== 5. RECOMPILE ==========
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

# ========== 6. CALLBACKS ==========
callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        "best_model_final.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
]

# ========== 7. TRAIN ==========
print("\n=== PHASE 2 FIXED: Fine-tuning top 120 layers ===")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=35,
    initial_epoch=10,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# ========== 8. FINAL RESULT ==========
model = tf.keras.models.load_model("best_model_final.keras")
loss, acc = model.evaluate(val_ds)
print(f"\n{'='*50}")
print(f"FINAL VALIDATION ACCURACY: {acc*100:.2f}%")
print(f"{'='*50}")

with open("class_names.txt", "w") as f:
    f.write("\n".join(class_names))