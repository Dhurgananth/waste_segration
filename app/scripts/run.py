#!/usr/bin/env python3

import os
import tensorflow as tf

# ------------------------------------------------------------
# MODEL PATH
# ------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "best_model.keras"
)

print("=" * 60)
print("EFFICIENTNET MODEL INSPECTION")
print("=" * 60)

print("\nModel path:")
print(MODEL_PATH)

# ------------------------------------------------------------
# CHECK MODEL
# ------------------------------------------------------------

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

print("\nModel file found.")
print("Loading model...")

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully!")

# ------------------------------------------------------------
# MODEL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)

model.summary()

# ------------------------------------------------------------
# INPUT / OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL INFORMATION")
print("=" * 60)

print("\nInput shape:")
print(model.input_shape)

print("\nOutput shape:")
print(model.output_shape)

print("\nTotal parameters:")
print(f"{model.count_params():,}")

# ------------------------------------------------------------
# FIND EFFICIENTNET
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("EFFICIENTNET INFORMATION")
print("=" * 60)

found = False

for layer in model.layers:

    if "efficientnet" in layer.name.lower():

        print(
            "\nLayer name:",
            layer.name
        )

        print(
            "Layer type:",
            layer.__class__.__name__
        )

        print(
            "Parameters:",
            f"{layer.count_params():,}"
        )

        found = True

if not found:
    print("\nNo EfficientNet layer found at top level.")

# ------------------------------------------------------------
# ALL LAYERS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("ALL MODEL LAYERS")
print("=" * 60)

for i, layer in enumerate(model.layers):

    print(
        f"{i:3d} | "
        f"{layer.name:40s} | "
        f"{layer.__class__.__name__}"
    )

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)