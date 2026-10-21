# 1. IMPORT LIBRARIES
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import (
    preprocess_input, decode_predictions
)

# 2. LOAD DATASET
path = "ImageNet_Dataset"

data = tf.keras.utils.image_dataset_from_directory(
    path,
    image_size=(224, 224),
    batch_size=8,
    shuffle=True
)

class_names = data.class_names
print("Classes:", class_names)

# 3. SELECT 5 CLASSES
selected_classes = class_names[:5]

print("Selected classes:", selected_classes)

# 4. LOAD PRETRAINED MODEL
model = ResNet50(weights="imagenet")

# 5. GET IMAGES AND PREPROCESS
images, labels = next(iter(data))

original_images = images.numpy().astype("uint8")
processed_images = preprocess_input(images.numpy())

print("Original shape:", original_images.shape)
print("Processed shape:", processed_images.shape)

# 6. PREDICT
predictions = model.predict(processed_images, verbose=0)

# 7. DISPLAY RESULTS
for i in range(len(original_images)):

    plt.imshow(original_images[i])
    plt.axis("off")
    plt.show()

    actual_label = class_names[int(labels[i])]
    print("Actual folder label:", actual_label)

    top5 = decode_predictions(
        predictions[i:i+1], top=5
    )[0]

    print("Top-5 ImageNet predictions:")

    for _, name, score in top5:
        print(name, ":", round(float(score) * 100, 2), "%")

    print("--------------------------")
