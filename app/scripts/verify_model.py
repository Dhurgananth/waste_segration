#!/usr/bin/env python3
import tensorflow as tf

print("Loading model...")
model = tf.keras.models.load_model("best_model.keras")

# Find base model
base_model = None
for layer in model.layers:
    if "efficientnet" in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    raise ValueError("Base model not found!")

print(f"\nBase model: {base_model.name}")
print(f"Total layers in base: {len(base_model.layers)}")

# Unfreeze top 120
base_model.trainable = True
for layer in base_model.layers[:-120]:
    layer.trainable = False

trainable = sum(1 for l in base_model.layers if l.trainable)
frozen = len(base_model.layers) - trainable
print(f"\nFrozen layers: {frozen}")
print(f"Trainable layers: {trainable}")

# Verify the head is also trainable
print(f"\nTop layer (Dense): trainable = {model.layers[-1].trainable}")
print(f"Dropout: trainable = {model.layers[-2].trainable}")

# Count total trainable parameters
total_trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
total_frozen = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
print(f"\nTotal trainable params: {total_trainable:,}")
print(f"Total frozen params: {total_frozen:,}")

if total_trainable > 0 and trainable > 0:
    print("\n✅ VERIFIED: Model is ready for fine-tuning")
else:
    print("\n❌ ERROR: Nothing is trainable!")