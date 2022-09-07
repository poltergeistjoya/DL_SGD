#!/bin/env python3.8

"""
SGD to fit sin wave to noisy sin wave points, modified from Chris Curro example assignment
We want to minimize y_hat est error (loss)= by finding ests for basis, weight, mean, and std dev
"""
import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags
from tqdm import trange

from dataclasses import dataclass, field, InitVar

#gives path of library being imported
script_path = os.path.dirname(os.path.realpath(__file__))

#use @dataclass implicityl makes __init__ and adds sick methods to class objects

@dataclass
class Data:

    #InitVar fields are psuedo-fields and only used by __init__ and __postinit__
    rng: InitVar[np.random.Generator]
    num_features: int
    num_samples: int
    sigma: float
    #field(init=False means parameter not needed when __init__ calde to nmake instance of class)
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    clean_y: np.ndarray = field(init=False)
    noise: np.ndarray= field(init=False)

    #after __init__ params __post_init__ updates other params
    def __post_init__(self, rng):
        #return evenly spaced values from 0 to num_samples
        self.index = np.arange(self.num_samples)

        #random numbers for uniform distribution
        #mult num_features becomes an error checking case
        #sample x from uniform distribution
        self.x = rng.uniform(0, 1, size=(self.num_samples, self.num_features))

        #clean_y are the y's without the noise added
        self.clean_y = np.sin(2*(np.pi)*self.x) #@ self.model.weights[:, np.newaxis] + self.model.bias
        clean_y = np.sin(2*(np.pi)*self.x)

        noise= rng.uniform(0, 0.1, size =(self.num_samples, self.num_features))
        #self-y is what picks the noisy points
        #why are we picking the noise points like this and not clean_y+noise
        self.y=rng.normal(loc=clean_y, scale = 0.1)
        #self.y = clean_y + noise
        #rng.normal(loc=clean_y, scale=self.sigma)

    def get_batch(self, rng, batch_size):
        """
        Select random subset of examples for training batch
        """
        choices = rng.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices].flatten()

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_features", 1, "Number of features in record")
flags.DEFINE_integer("num_samples", 50, "Number of samples in dataset")
flags.DEFINE_integer("batch_size", 16, "Number of samples in batch")
flags.DEFINE_integer("num_iters", 300, "Number of SGD iterations")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate / step size for SGD")
flags.DEFINE_integer("random_seed", 31415, "Random seed")
flags.DEFINE_float("sigma_noise", 0.5, "Standard deviation of noise random variable")
flags.DEFINE_bool("debug", False, "Set logging level to debug")
#make flag for number of basis functions
flags.DEFINE_integer("num_basis", 3, "Number of basis functions aka M")

#this does the estimations and creates variables to be trained
class Model(tf.Module):
    def __init__(self, rng, num_features, num_basis):
        """
        A plain linear regression model with a bias term
        """
        self.num_basis= num_basis
        self.num_features = num_features
        #need tf.Variable to make it a tunable variable
        self.w = tf.Variable(rng.normal(shape=[num_basis, 1]))
        self.b = tf.Variable(tf.zeros(shape=[1, 1]))
        #add mu and sigma as they are also parameters for the estimation
        self.mu = tf.Variable(rng.normal(shape=[1,self.num_basis]))
        self.sigma = tf.Variable(rng.normal(shape=[1,self.num_basis]))

    #__call__ to make y hat
    def __call__(self, x):
        phi = tf.math.exp((-(x-self.mu)**2)/(self.sigma**2))
        return tf.squeeze(phi @ self.w + self.b)

    @property
    def model(self):
        return LinearModel(
            self.w.numpy(), self.b.numpy().squeeze(), self.mu, self.sigma
        )


def main(a):
    logging.basicConfig()

    if FLAGS.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Safe np and tf PRNG
    seed_sequence = np.random.SeedSequence(FLAGS.random_seed)
    np_seed, tf_seed = seed_sequence.spawn(2)
    np_rng = np.random.default_rng(np_seed)
    tf_rng = tf.random.Generator.from_seed(tf_seed.entropy)

    data = Data(
        np_rng,
        FLAGS.num_features,
        FLAGS.num_samples,
        FLAGS.sigma_noise,
    )

    model = Model(tf_rng, FLAGS.num_features, FLAGS.num_basis)

    #this is what does the SGD at the specified learning rate
    optimizer = tf.optimizers.SGD(learning_rate=FLAGS.learning_rate)

    bar = trange(FLAGS.num_iters)
    #autodiff in batches, GradientTape is what does the autodiff
    for i in bar:
        with tf.GradientTape() as tape:
            x, y = data.get_batch(np_rng, FLAGS.batch_size)
            y_hat = model(x)
            loss = 0.5 * tf.reduce_mean((y_hat - y) ** 2)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()


#PLOTTING

    if FLAGS.num_features > 1:
        # Only continue to plotting if x is a scalar
        exit(0)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)

    ax[0].set_title("Sine fit with SGD")
    ax[0].set_xlabel("x")
    ax[0].set_ylim(np.amax(data.clean_y) * -1.5, np.amax(data.clean_y) * 1.5)
    h = ax[0].set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    xs = np.linspace(0, 1, 100)
    xs = xs[:, np.newaxis]
    ax[0].plot(np.squeeze(xs), np.sin(2*np.pi*xs), "-", np.squeeze(data.x), data.y, "o", np.squeeze(xs), np.squeeze(model(xs)), "--")

    ax[1].set_title("Gaussian Basis Functions")
    ax[1].set_xlabel("x")
    ax[1].set_ylim(0, 1.5)
    h = ax[1].set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    x2 = np.linspace(-4, 4, 400)
    x2 = x2[:, np.newaxis]
    phi = tf.math.exp((-(x2-model.mu)**2)/(model.sigma**2))
    ax[1].plot(np.squeeze(x2), np.squeeze(phi), "-")


    plt.tight_layout()
    plt.savefig(f"{script_path}/sinfit.pdf")


if __name__ == "__main__":
    app.run(main)
