# from keras.models import load_model
# import numpy as np
# import pandas as pd
# import os
# import matplotlib.pyplot as plt
# from keras.models import Sequential
# from keras.layers import Dense
# from numpy.random import randn
# from matplotlib import pyplot
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import metrics
# data = pd.read_csv(
#     r"D:\Stuff\CyberSec\Datasets\IDS2018\02-14-2018_processed.csv")
# print(data.shape)
# print(data.tail())
# print(data.columns)

# data = data.dropna()
# X = data.drop(["Label"], axis=1)
# y = data["Label"]
# n_cols = data.shape[1]


# def generate_latent_points(latent_dim, n_samples):
#     x_input = randn(latent_dim * n_samples)
#     x_input = x_input.reshape(n_samples, latent_dim)
#     return x_input


# def generate_fake_samples(generator, latent_dim, n_samples):
#     x_input = generate_latent_points(latent_dim, n_samples)
#     X = generator.predict(x_input)
#     y = np.zeros((n_samples, 1))

#     return X, y


# def generate_real_samples(n):
#     X = data.sample(n)
#     y = np.ones((n, 1))
#     return X, y


# def define_generator(latent_dim, n_outputs=n_cols):
#     model = Sequential()
#     model.add(Dense(15, activation='relu',
#               kernel_initializer='he_uniform', input_dim=latent_dim))
#     model.add(Dense(30, activation='relu'))
#     model.add(Dense(n_outputs, activation='linear'))
#     return model


# generator1 = define_generator(10, n_cols)
# generator1.summary()


# def define_discriminator(n_inputs=n_cols):
#     model = Sequential()
#     model.add(Dense(25, activation='relu',
#               kernel_initializer='he_uniform', input_dim=n_inputs))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam', metrics=['accuracy'])
#     return model


# discriminator1 = define_discriminator(n_cols)
# discriminator1.summary()


# def define_gan(generator, discriminator):
#     discriminator.trainable = False
#     model = Sequential()
#     model.add(generator)
#     model.add(discriminator)
#     model.compile(loss='binary_crossentropy', optimizer='adam')
#     return model


# def plot_history(d_hist, g_hist):
#     plt.subplot(1, 1, 1)
#     plt.plot(d_hist, label='d')
#     plt.plot(g_hist, label='gen')
#     plt.show()
#     plt.close()


# def train(g_model, d_model, gan_model, latent_dim, n_epochs=100, n_batch=128, n_eval=200):
#     half_batch = int(n_batch / 2)
#     d_history = []
#     g_history = []
#     for epoch in range(n_epochs):
#         x_real, y_real = generate_real_samples(half_batch)
#         x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
#         d_loss_real, d_real_acc = d_model.train_on_batch(x_real, y_real)
#         d_loss_fake, d_fake_acc = d_model.train_on_batch(x_fake, y_fake)
#         d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
#         x_gan = generate_latent_points(latent_dim, n_batch)
#         y_gan = np.ones((n_batch, 1))
#         g_loss_fake = gan_model.train_on_batch(x_gan, y_gan)
#         print('>%d, d1=%.3f, d2=%.3f d=%.3f g=%.3f' %
#               (epoch+1, d_loss_real, d_loss_fake, d_loss,  g_loss_fake))
#         d_history.append(d_loss)
#         g_history.append(g_loss_fake)
#         # plot_history(d_history, g_history)
#         g_model.save('trained_generated_model.h5')


# latent_dim = 10
# discriminator = define_discriminator()
# generator = define_generator(latent_dim)
# gan_model = define_gan(generator, discriminator)
# train(generator, discriminator, gan_model, latent_dim)

# model = load_model("trained_generated_model.h5")

# latent_points = generate_latent_points(10, 5000)
# X = model.predict(latent_points)

# column_names = list(data.columns)

# data_fake = pd.DataFrame(data=X,  columns=column_names)
# data_fake.head()

# outcome_mean = data_fake.Label.mean()
# data_fake['Label'] = data_fake['Label'] > outcome_mean
# data_fake["Label"] = data_fake["Label"].astype(int)

# column_names = list(data.columns)
# column_names.pop()

# features = column_names
# label = ['Label']
# X_fake_created = data_fake[features]
# y_fake_created = data_fake[label]

# data_fake.to_csv('synthetic_data_10L.csv', index=False)


from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, UpSampling1D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess the real data
real_data = pd.read_csv(
    r"D:\Stuff\CyberSec\Datasets\IDS2018\02-14-2018_processed.csv")
scaler = MinMaxScaler()
scaled_real_data = scaler.fit_transform(real_data)

# Define the generator model


# def build_generator(latent_dim):
#     model = Sequential()
#     model.add(Dense(64, input_dim=latent_dim))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Dense(32))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Dense(real_data.shape[1], activation='sigmoid'))
#     return model


def build_generator(latent_dim, output_shape):
    model = Sequential()

    # Add a dense layer with input_dim equal to the latent dimension
    model.add(Dense(10, input_dim=latent_dim, activation='relu'))

    # Add a reshape layer to transform the 10 outputs into a 10x1 tensor
    model.add(Reshape((10, 1)))

    # Add a convolutional layer with kernel size 3 and 32 filters
    model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))

    # Add another convolutional layer with kernel size 3 and 16 filters
    model.add(Conv1D(16, kernel_size=3, padding='same', activation='relu'))

    # Add a flatten layer to convert the output to a 2D tensor
    model.add(Flatten())

    # Add a dense layer with output_shape neurons and sigmoid activation
    model.add(Dense(np.prod(output_shape), activation='sigmoid'))

    # Reshape the output tensor to the desired output_shape
    model.add(Reshape(output_shape))

    return model


# Define the discriminator model


def build_discriminator():
    model = Sequential()
    model.add(Dense(32, input_shape=(real_data.shape[1],)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(16))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Define the GAN model


def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    optimizer = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

# Define the function to generate fake samples


def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = np.random.randn(
        latent_dim * n_samples).reshape(n_samples, latent_dim)
    x_fake = generator.predict(x_input)
    y_fake = np.zeros((n_samples, 1))
    return x_fake, y_fake

# Define the function to train the GAN


def train_gan(generator, discriminator, gan, scaled_real_data, latent_dim, n_epochs=10000, n_batch=32, n_eval=1000):
    for i in range(n_epochs):
        # Train discriminator
        for j in range(n_batch):
            x_real = scaled_real_data[np.random.randint(
                0, scaled_real_data.shape[0], n_batch), :]
            y_real = np.ones((n_batch, 1))
            x_fake, y_fake = generate_fake_samples(
                generator, latent_dim, n_batch)
            discriminator_loss_real = discriminator.train_on_batch(
                x_real, y_real)
            discriminator_loss_fake = discriminator.train_on_batch(
                x_fake, y_fake)
            discriminator_loss = 0.5 * \
                np.add(discriminator_loss_real, discriminator_loss_fake)
        # Train generator
        x_gan = np.random.randn(
            latent_dim * n_batch).reshape(n_batch, latent_dim)
        y_gan = np.ones((n_batch, 1))
        generator_loss = gan.train_on_batch(x_gan, y_gan)
        # Evaluate and print progress
        if i % n_eval == 0:
            print(
                f'Epoch {i}, Generator Loss: {generator_loss}, Discriminator Loss: {discriminator_loss}')


# Define the latent dimension
latent_dim = 100

# Define the number of epochs and batch size for training
# n_epochs = 1000
# n_batch = 32
# n_eval = 1000
n_epochs = 10
n_batch = 10
n_eval = 200

# Build the generator, discriminator, and GAN models
generator = build_generator(latent_dim, real_data.shape[1:])
discriminator = build_discriminator()
optimizer = Adam(lr=0.0002, beta_1=0.5)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
gan = build_gan(generator, discriminator)

# Train the GAN
train_gan(generator, discriminator, gan, scaled_real_data,
          latent_dim, n_epochs, n_batch, n_eval)

# Generate synthetic data
# n_samples = real_data.shape[0]
n_samples = 750
x_synthetic, y_synthetic = generate_fake_samples(
    generator, latent_dim, n_samples)
unscaled_synthetic_data = scaler.inverse_transform(x_synthetic)
synthetic_data = pd.DataFrame(
    unscaled_synthetic_data, columns=real_data.columns)

# Save synthetic data to a CSV file
synthetic_data.to_csv('synthetic_data.csv', index=False)
