# EXPERIMENT 4: IMAGE CAPTION GENERATION USING CNN AND LSTM

# 1. IMPORT LIBRARIES
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, RepeatVector, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# 2. LOAD DATASET
path = "ImageNet_Dataset"

data = tf.keras.utils.image_dataset_from_directory(
    path, image_size=(224, 224), batch_size=16,
    shuffle=False
)

class_names = data.class_names
print("Classes:", class_names)

# 3. LOAD PRETRAINED CNN
cnn = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)
cnn.trainable = False

# 4. EXTRACT IMAGE FEATURES
features = []
labels = []
sample_images = []

for images, batch_labels in data:
    processed = preprocess_input(images.numpy())
    f = cnn.predict(processed, verbose=0)

    features.extend(f)
    labels.extend(batch_labels.numpy())
    sample_images.extend(images.numpy().astype("uint8"))

features = np.array(features)
labels = np.array(labels)
sample_images = np.array(sample_images)

print("Feature shape:", features.shape)

# 5. PREPARE CAPTIONS
captions = [
    "startseq a photo of a " +
    class_names[int(label)].replace("_", " ").replace("-", " ") +
    " endseq"
    for label in labels
]

tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(captions)

sequences = tokenizer.texts_to_sequences(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(seq) for seq in sequences)

print("Vocabulary size:", vocab_size)
print("Maximum caption length:", max_length)

# 6. CREATE INPUT AND TARGET WORDS
X_text = []
X_features = []
y_words = []

for i, seq in enumerate(sequences):
    for j in range(1, len(seq)):
        X_text.append(seq[:j])
        X_features.append(features[i])
        y_words.append(seq[j])

X_text = pad_sequences(
    X_text, maxlen=max_length - 1, padding="pre"
)
X_features = np.array(X_features)
y_words = np.array(y_words)

# 7. SPLIT DATA
Xf_train, Xf_test, Xt_train, Xt_test, y_train, y_test = (
    train_test_split(
        X_features, X_text, y_words,
        test_size=0.2, random_state=42
    )
)

# 8. BUILD CNN-LSTM MODEL
image_input = Input(shape=(2048,))
text_input = Input(shape=(max_length - 1,))

img_features = Dense(128, activation="relu")(image_input)
img_features = RepeatVector(max_length - 1)(img_features)

text_features = Embedding(
    vocab_size, 128
)(text_input)

combined = Concatenate()([img_features, text_features])
lstm = LSTM(128)(combined)
output = Dense(vocab_size, activation="softmax")(lstm)

model = Model([image_input, text_input], output)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 9. TRAIN MODEL
model.fit(
    [Xf_train, Xt_train], y_train,
    epochs=10, batch_size=32, verbose=1
)

# 10. GENERATE CAPTION
def generate_caption(feature):
    caption = ["startseq"]

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences(
            [" ".join(caption)]
        )
        seq = pad_sequences(
            seq, maxlen=max_length - 1, padding="pre"
        )

        pred = model.predict(
            [feature.reshape(1, 2048), seq],
            verbose=0
        )

        word_id = int(np.argmax(pred[0]))
        word = tokenizer.index_word.get(word_id, "")

        if word == "endseq" or word == "":
            break

        caption.append(word)

    return " ".join(caption[1:])

# 11. TEST AND DISPLAY CAPTION
test_index = 0

plt.imshow(sample_images[test_index])
plt.axis("off")
plt.show()

print("Actual class:", class_names[labels[test_index]])
print("Expected caption:", captions[test_index])
print("Generated caption:", generate_caption(features[test_index]))
