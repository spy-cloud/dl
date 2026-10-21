# EXPERIMENT 3: FEATURE EXTRACTION AND IMAGE CLASSIFICATION

# 1. IMPORT LIBRARIES
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

# 2. LOAD DATASET
path = "ImageNet_Dataset"

data = tf.keras.utils.image_dataset_from_directory(
    path,
    image_size=(224, 224),
    batch_size=8,
    shuffle=False
)

class_names = data.class_names
print("Classes:", class_names)

# 3. LOAD PRETRAINED CNN
model = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

model.trainable = False

# 4. EXTRACT FEATURES
features = []
labels = []

for images, batch_labels in data:

    images = preprocess_input(images.numpy())

    batch_features = model.predict(images, verbose=0)

    features.extend(batch_features)
    labels.extend(batch_labels.numpy())

features = np.array(features)
labels = np.array(labels)

print("Feature shape:", features.shape)
print("Labels shape:", labels.shape)

# 5. SPLIT INTO TRAINING AND TESTING DATA
X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# 6. TRAIN CLASSIFIER
classifier = LogisticRegression(max_iter=1000)

classifier.fit(X_train, y_train)

# 7. PREDICT
y_pred = classifier.predict(X_test)

# 8. EVALUATE MODEL
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(
    y_test, y_pred, average="macro", zero_division=0
))

print("Recall:", recall_score(
    y_test, y_pred, average="macro", zero_division=0
))

print("F1 Score:", f1_score(
    y_test, y_pred, average="macro", zero_division=0
))

# 9. DISPLAY CONFUSION MATRIX
cm = confusion_matrix(
    y_test, y_pred, labels=np.arange(len(class_names))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(cmap="Blues", xticks_rotation=90)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
