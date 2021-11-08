# From Jason Brownlee:
# https://github.com/PDillis/generative-zoo/blob/master/gan/1dgan/keras/1dGAN.py
# https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-1-dimensional-function-from-scratch-in-keras/

import matplotlib
import matplotlib.pyplot as plt
import imageio

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def f(x):
    return np.tan(x)


def define_discriminator(n_inputs=2):
    model = Sequential()
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


#### Train the discriminator ####
# def train_discriminator(model, n_epochs=1000, n_batch=128):
#     half_batch = int(n_batch / 2)
#     # Run the epochs manually
#     for i in range(n_epochs):
#         # Generate real examples
#         X_real, y_real = generate_real_samples(half_batch)
#         model.train_on_batch(X_real, y_real)
#         # Generate fake examples
#         X_fake, y_fake = generate_fake_samples(half_batch)
#         model.train_on_batch(X_fake, y_fake)
#         # Evaluate the model
#         _, acc_real = model.evaluate(X_real, y_real, verbose=0)
#         _, acc_fake = model.evaluate(X_fake, y_fake, verbose=0)
#         print('{}/{} ... Acc real={:.3f} ... Acc fake={:.3f}'.format(i+1, n_epochs, acc_real, acc_fake))


def define_generator(latent_dim, n_outputs=2):
    model = Sequential()
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    model.add(Dense(n_outputs, activation='linear'))
    return model


def define_gan(generator, discriminator):
    # Make weights in the discriminator not trainable
    discriminator.trainable = False
    # Connect them
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def generate_real_samples(n=100):
    # Generate inputs in [-1.5, 1.5]
    X1 = 3 * (np.random.rand(n) - 0.5)
    # Get the outputs f(x)
    X2 = f(X1)
    # Stack the arrays
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = np.hstack((X1, X2))
    # Now generate the class labels
    Y = np.ones((n, 1))
    return X, Y


def generate_latent_points(latent_dim, n):
    # Generate points in the latent space
    x_input = np.random.randn(latent_dim * n)
    # Reshape into a batch of inputs for the network
    x_input = x_input.reshape(n, latent_dim)
    return x_input


def generate_fake_samples(generator, latent_dim, n):
    # Generate points in the latent space
    x_input = generate_latent_points(latent_dim, n)
    # Predict the outputs
    X = generator.predict(x_input)
    # Create class labels
    Y = np.zeros((n, 1))
    return X, Y


# Plot real and fake points
def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
    # Prepare real samples
    x_real, y_real = generate_real_samples(n)
    # Evaluate the discriminator on real samples
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    # Prepare fake samples
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
    # Evaluate the discriminator on fake samples
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    # Scatter plot real and fake samples
    ax.scatter(x_real[:, 0], x_real[:, 1], color='red')
    ax.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
    ax.set(title='Epoch: {}... Acc real={:.3f} ... Acc fake={:.3f}'.format(epoch + 1, acc_real, acc_fake))

    # Animation code
    ax.set_ylim(-15, 15)
    ax.set_xlim(-2, 2)
    # Draw the canvas, cache the renderer
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image


def train(g_model, d_model, gan_model, latent_dim, fr, n_epochs=1000, n_batch=128):
    # Determine half the size of one batch, for updating the discriminator
    half_batch = int(n_batch / 2)
    # Manually enumerate epochs
    for i in range(n_epochs):
        print("开始第 {} 轮训练...".format(i))
        # Prepare real samples
        x_real, y_real = generate_real_samples(half_batch)
        # Prepare fake samples
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # We update the discriminator
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)
        # Prepare points in latent space as input for the generator
        x_gan = generate_latent_points(latent_dim, n_batch)
        # Create inverted labels for the fake samples
        y_gan = np.ones((n_batch, 1))
        # Update the generator via the discriminator error
        gan_model.train_on_batch(x_gan, y_gan)

        if i % 10 == 0:
            fr.append(summarize_performance(i, generator, discriminator, latent_dim))
        # Evaluate the model every n_eval epochs:
        # if (i+1) % n_eval == 0:
        #     summarize_performance(i, g_model, d_model, latent_dim)


frames = []

kwargs_write = {'fps': 30.0, 'quantizer': 'nq'}

latent_dim = 5
n_epochs = 10000

discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan_model = define_gan(generator, discriminator)

if __name__ == '__main__':
    train(g_model=generator, d_model=discriminator, gan_model=gan_model, latent_dim=latent_dim, fr=frames)
    imageio.mimsave('./tan.avi', frames, fps=60)
