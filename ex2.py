# EXPERIMENT 2: FILTERS AND FEATURE MAPS

# 1. IMPORT LIBRARIES
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# 2. LOAD DATASET
path = "ImageNet_Dataset"

data = tf.keras.utils.image_dataset_from_directory(
    path, image_size=(224, 224), batch_size=8
)

class_names = data.class_names
print("Classes:", class_names)

# 3. LOAD PRETRAINED MODEL
model = VGG16(
    weights="imagenet",
    include_top=False
)

# 4. GET AND PREPROCESS IMAGES
images, labels = next(iter(data))

original_images = images.numpy().astype("uint8")
processed_images = preprocess_input(images.numpy())

print("Image shape:", processed_images.shape)

# 5. SELECT CONVOLUTIONAL LAYERS
layers = [
    "block1_conv1",
    "block3_conv1",
    "block5_conv1"
]

outputs = [model.get_layer(n).output for n in layers]

feature_model = tf.keras.Model(
    inputs=model.input, outputs=outputs
)

# 6. EXTRACT FEATURE MAPS
features = feature_model.predict(processed_images, verbose=0)

# 7. DISPLAY ORIGINAL IMAGE
plt.imshow(original_images[0])
plt.title(class_names[int(labels[0])])
plt.axis("off")
plt.show()

# 8. DISPLAY FEATURE MAPS
for name, feature in zip(layers, features):

    print(name, feature.shape)

    plt.figure(figsize=(12, 4))

    for i in range(8):
        plt.subplot(1, 8, i + 1)
        plt.imshow(feature[0, :, :, i], cmap="gray")
        plt.axis("off")

    plt.suptitle(name)
    plt.show()

# 9. DISPLAY CONVOLUTIONAL FILTERS
filters, biases = model.get_layer("block1_conv1").get_weights()

plt.figure(figsize=(8, 4))

for i in range(8):
    f = filters[:, :, :, i]
    f = (f - f.min()) / (f.max() - f.min() + 1e-8)

    plt.subplot(2, 4, i + 1)
    plt.imshow(f)
    plt.axis("off")

plt.suptitle("Convolutional Filters")
plt.show()
