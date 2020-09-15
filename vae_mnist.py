#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 14:48:09 2020

@author: pasq
"""
from IPython import display
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  #return np.where(images > .5, 1.0, 0.0).astype('float32')
  return images.astype("float32")

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

train_size = 60000
batch_size = 32
test_size = 10000


train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder"""    

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        
        #Encoder network definition
        self.encoder = tf.keras.Sequential(
                [
                        tf.keras.layers.InputLayer(input_shape=(28,28,1)),
                        tf.keras.layers.Conv2D(
                                filters=32, kernel_size=3, strides=(2,2), activation="relu"),
                        tf.keras.layers.Conv2D(
                                filters=64, kernel_size=3, strides=(2,2), activation="relu"),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(self.latent_dim*2)
                        ]
                )
                
        #Decoder network definition
        self.decoder = tf.keras.Sequential(
                [
                        tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                        tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
                        tf.keras.layers.Reshape(target_shape=(7,7,32)),
                        tf.keras.layers.Conv2DTranspose(
                                filters=64, kernel_size=3, strides=2, padding="same", activation="relu"),
                        tf.keras.layers.Conv2DTranspose(
                                filters=32, kernel_size=3, strides=2, padding="same", activation="relu"),
                        tf.keras.layers.Conv2DTranspose(
                                filters=1, kernel_size=3, strides=1, padding="same")
                        ]
                )
        
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        
        return self.decode(eps, apply_sigmoid=True)
    
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar
    
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        
        return logits
    

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

    
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    
    #reparametrization trick below
    eps = tf.random.normal(shape=mean.shape)
    z = eps*tf.exp(logvar*.5)+mean
    x_logit = model.decode(z)
    
    #by definition we use a cross_entropy for logits of the 0-9 classification problem
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
optimizer = tf.keras.optimizers.Adam(1e-4)

epochs = 3
latent_dim = 2
model = CVAE(latent_dim)
num_examples_to_generate = 16

def generate_and_save_images(model, epoch, test_sample):
  mean, logvar = model.encode(test_sample)
  eps = tf.random.normal(shape=mean.shape)
  z = eps*tf.exp(logvar*.5)+mean
  predictions = model.sample(z)
  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions_3.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions_3[i, :, :, 0])
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()


# Pick a sample of the test set for generating output images
assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
  test_sample = test_batch[0:num_examples_to_generate, :, :, :]




#for epoch in range(1, epochs + 1):
#  for train_x in train_dataset:
#    train_step(model, train_x, optimizer)
#
#  loss = tf.keras.metrics.Mean()
#  for test_x in test_dataset:
#    loss(compute_loss(model, test_x))
#  elbo = -loss.result()
#  print('Epoch: {}, Test set ELBO: {}'
#        .format(epoch, elbo))
#

for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    train_step(model, train_x, optimizer)
  end_time = time.time()

  loss = tf.keras.metrics.Mean()
  for test_x in test_dataset:
    compute_loss(model, test_x)
  elbo = -loss.result()
  display.clear_output(wait=False)
  print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))
  generate_and_save_images(model, epoch, test_sample)
      

#Sampling for a single number
k = 10
mean, logvar = model.encode(test_sample)
eps = tf.random.normal(shape=mean.shape)
z_3 = eps*tf.exp(logvar[k,:]*.5)+mean[k,:]
predictions_3 = model.sample(z_3)
fig = plt.figure(figsize=(4,4))

for i in range(predictions_3.shape[0]):
  plt.subplot(4, 4, i + 1)
  plt.imshow(predictions_3[i, :, :, 0], cmap='gray')
  plt.axis('off')
  
plt.imshow(test_sample[k,:,:,0])

