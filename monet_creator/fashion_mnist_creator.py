# Bringing in tensorflow
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import BinaryCrossentropy

import os
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback

# Brining in tensorflow datasets for fashion mnist
import tensorflow_datasets as tfds
# Bringing in matplotlib for viz stuff
from matplotlib import pyplot as plt


# Data handling
def load_data():
    return tfds.load('fashion_mnist', split='train')

def preprocess_data(data):
    image = data['image']
    image = image / 255
    return image

def create_ds(ds):
    ds = ds.map(preprocess_data)
    ds = ds.cache()
    ds = ds.batch(180)
    ds = ds.prefetch(64)
    return ds

# GAN
def build_generator():
    model = Sequential()

    # Input block
    model.add(Dense(7*7*128,input_dim = 128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7,7,128)))

    # Upsampling block 1
    model.add(UpSampling2D())
    model.add(Conv2D(128,5,padding='same'))
    model.add(LeakyReLU(0.2))

    # Upsampling block 2
    model.add(UpSampling2D())
    model.add(Conv2D(128,5,padding='same'))
    model.add(LeakyReLU(0.2))

    # Conv block 1
    model.add(Conv2D(128,5,padding='same'))
    model.add(LeakyReLU(0.2))

    # Conv block 2
    model.add(Conv2D(128,4,padding='same'))
    model.add(LeakyReLU(0.2))

    # Output block
    model.add(Conv2D(1,4,padding='same'))

    return model

def build_discriminator():
    model = Sequential()

    # Input block
    model.add(Conv2D(32,5,padding='valid',input_shape=(28,28,1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Downsampling block 1
    model.add(Conv2D(64,5,padding='valid',input_shape=(28,28,1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Downsampling block 2
    model.add(Conv2D(128,5,padding='valid',input_shape=(28,28,1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

     # Downsampling block 3
    model.add(Conv2D(128,5,padding='valid',input_shape=(28,28,1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Final block
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    return model

class FashGan(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args,**kwargs)

        self.generator = generator
        self.discriminator = discriminator()

    def compile(self,g_opt,d_opt,g_loss,d_loss, *args, **kwargs):
        super().compile(*args,**kwargs)

        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_step(self, batch):
        random_input = tf.random.uniform((128,128,1))
        fake_images = self.generator(random_input, training=False)
        real_images = batch

        # Train discriminator
        with tf.GradientTape() as d_tape :
            y_hat_fake = self.discriminator(fake_images,training=True)
            y_hat_real = self.discriminator(real_images,training=True)
            y_fake = tf.ones_like(y_hat_fake) - 0.15
            y_real = tf.zeros_like(y_hat_real) + 0.15
            y_hat_tot = tf.concat([y_hat_fake,y_hat_real],axis=0)
            y_tot = tf.concat([y_fake,y_real],axis=0)

            discriminator_loss = self.d_loss(y_hat_tot,y_tot)
        d_grad = d_tape.gradient(discriminator_loss,self.discriminator.trainable_variables)
        self.d_opt.apply_gradient(zip(d_grad,self.discriminator.trainable_variables))

        # Train generator
        with tf.GradientTape() as g_tape :
            random_input = tf.random.uniform((128,128,1))
            fake_images = self.generator(random_input,training=True)

            y_hat_fake = self.discriminator(fake_images,training=False)
            y_fake = tf.zeros_like(y_hat_fake) # inverse since we want the generator to be able to fool the discriminator
            generator_loss = self.g_loss(y_hat_fake,y_fake)
        g_grad = g_tape.gradient(generator_loss,self.generator.trainable_variables)
        self.g_opt.apply_gradient(zip(g_grad,self.generator.trainable_variables))

        return {'Discriminator loss :':discriminator_loss,'Generator loss :':generator_loss}

class ModelMonitor(Callback):
    def  __init__(self, num_img = 3, latent_space = 128):
        self.num_img = num_img
        self.latent_space = latent_space

    def on_epoch_end(self, epoch, logs = None):
        random_latent_vectors = tf.random.uniform((self.num_img,self.latent_space))
        imgs = self.model.generator(random_latent_vectors)
        imgs *= 255
        imgs = imgs.numpy()
        for i in range(self.num_img):
            img = array_to_img(imgs[i])
            img.save(os.path.join('images', f'generated_img_{epoch}_{i}.png'))

if __name__ == "__main__":

    data = load_data()
    ds = preprocess_data(data)

    generator = build_generator()
    discriminator = build_discriminator()
    g_opt = Adam(learning_rate = 0.0001)
    d_opt = Adam(learning_rate = 0.00001)
    g_loss = BinaryCrossentropy()
    d_loss = BinaryCrossentropy()

    my_gan = FashGan(generator,discriminator)
    my_gan.compile(g_opt,d_opt,g_loss,d_loss)

    hist = my_gan.fit(ds, epochs = 2000, callbacks = [ModelMonitor()])

    my_gan.save('my_model.h5')
