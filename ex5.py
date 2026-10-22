# EXPERIMENT 5: TEXT-TO-IMAGE SYNTHESIS USING CGAN

# 1. IMPORT LIBRARIES
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model

# 2. LOAD DATASET
path = "ImageNet_Dataset"

data = tf.keras.utils.image_dataset_from_directory(
    path,
    image_size=(32, 32),
    batch_size=32
)

class_names = data.class_names
num_classes = len(class_names)

print("Classes:", class_names)

# Normalize images to [-1, 1]
data = data.map(
    lambda x, y: (tf.cast(x, tf.float32) / 127.5 - 1, y)
)

# 3. SET PARAMETERS
noise_dim = 100
embed_dim = 32

# 4. BUILD GENERATOR
noise = layers.Input(shape=(noise_dim,))
label = layers.Input(shape=(), dtype="int32")

text = layers.Embedding(num_classes, embed_dim)(label)

x = layers.Concatenate()([noise, text])
x = layers.Dense(8 * 8 * 128, activation="relu")(x)
x = layers.Reshape((8, 8, 128))(x)

x = layers.UpSampling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

x = layers.UpSampling2D()(x)
image = layers.Conv2D(
    3, 3, padding="same", activation="tanh"
)(x)

generator = Model([noise, label], image)

# 5. BUILD DISCRIMINATOR
image_input = layers.Input(shape=(32, 32, 3))
label_input = layers.Input(shape=(), dtype="int32")

text = layers.Embedding(num_classes, 32)(label_input)
text = layers.Dense(32 * 32)(text)
text = layers.Reshape((32, 32, 1))(text)

x = layers.Concatenate()([image_input, text])
x = layers.Conv2D(64, 4, strides=2, padding="same")(x)
x = layers.LeakyReLU(negative_slope=0.2)(x)
x = layers.Conv2D(128, 4, strides=2, padding="same")(x)
x = layers.LeakyReLU(negative_slope=0.2)(x)
x = layers.Flatten()(x)

output = layers.Dense(1)(x)

discriminator = Model([image_input, label_input], output)

# 6. DEFINE LOSS AND OPTIMIZERS
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

g_optimizer = tf.keras.optimizers.Adam(0.0002)
d_optimizer = tf.keras.optimizers.Adam(0.0002)

# 7. TRAIN GAN
@tf.function
def train_step(real_images, labels):

    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal((batch_size, noise_dim))

    with tf.GradientTape() as g_tape, \
         tf.GradientTape() as d_tape:

        fake_images = generator(
            [noise, labels], training=True
        )

        real_output = discriminator(
            [real_images, labels], training=True
        )

        fake_output = discriminator(
            [fake_images, labels], training=True
        )

        g_loss = loss_fn(
            tf.ones_like(fake_output), fake_output
        )

        d_loss = (
            loss_fn(tf.ones_like(real_output), real_output)
            + loss_fn(tf.zeros_like(fake_output), fake_output)
        )

    g_grad = g_tape.gradient(
        g_loss, generator.trainable_variables
    )

    d_grad = d_tape.gradient(
        d_loss, discriminator.trainable_variables
    )

    g_optimizer.apply_gradients(
        zip(g_grad, generator.trainable_variables)
    )

    d_optimizer.apply_gradients(
        zip(d_grad, discriminator.trainable_variables)
    )

    return g_loss, d_loss

# 8. RUN TRAINING
for epoch in range(5):

    g_losses = []
    d_losses = []

    for images, labels in data:
        g_loss, d_loss = train_step(images, labels)
        g_losses.append(float(g_loss))
        d_losses.append(float(d_loss))

    print(
        "Epoch:", epoch + 1,
        "Generator loss:", np.mean(g_losses),
        "Discriminator loss:", np.mean(d_losses)
    )

# 9. GENERATE IMAGES
def generate_images(class_id):

    noise = tf.random.normal((5, noise_dim))
    labels = tf.constant([class_id] * 5, dtype=tf.int32)

    generated = generator(
        [noise, labels], training=False
    ).numpy()

    generated = (generated + 1) / 2

    plt.figure(figsize=(10, 3))

    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(np.clip(generated[i], 0, 1))
        plt.axis("off")

    plt.suptitle("Generated: " + class_names[class_id])
    plt.show()

# Generate images for the first class
generate_images(0)
