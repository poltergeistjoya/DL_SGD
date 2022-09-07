#!/bin/env python3.8

"""
we want to minimize y_hat est error (loss)= by finding ests for basis, weight, mean, and std dev
"""

"""
Example assignment. Author: Chris Curro
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
class LinearModel:
    weights: np.ndarray
    bias: float


@dataclass
class Data:
    model: LinearModel

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
        clean_y = np.sin(2*(np.pi)*self.x) #@ self.model.weights[:, np.newaxis] + self.model.bias

        noise= rng.uniform(0, 0.1, size =(self.num_samples, self.num_features))
        #self-y is what picks the noisy points
        #why are we picking the noise points like this and not clean_y+noise
        self.y=rng.normal(loc=clean_y, scale = 0.1)
        #self.y = clean_y + noise
        #rng.normal(loc=clean_y, scale=self.sigma)

        for i in range(50):
            print(self.x[i],clean_y[i], self.y[i], noise[i])

        print(type(self.x[1]), type(clean_y[1]), type(self.y[1]),type(noise[1]))

    def get_batch(self, rng, batch_size):
        """
        Select random subset of examples for training batch
        """
        choices = rng.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices].flatten()


def compare_linear_models(a: LinearModel, b: LinearModel):
    for w_a, w_b in zip(a.weights, b.weights):
        print(f"{w_a:0.2f}, {w_b:0.2f}")

    print(f"{a.bias:0.2f}, {b.bias:0.2f}")


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
flags.DEFINE_integer("num_basis", 1, "Number of basis functions aka M")

#this does the estimations and creates variables to be trained
class Model(tf.Module):
    def __init__(self, rng, num_features):
        """
        A plain linear regression model with a bias term
        """
        self.num_features = num_features
        #need tf.Variable to make it a tunable variable
        self.w = tf.Variable(rng.normal(shape=[1, 1]))
        self.b = tf.Variable(tf.zeros(shape=[1, 1]))
        #add mu and sigma as they are also parameters for the estimation
        self.mu = tf.Variable(rng.normal(shape=[self.num_features, 1]))
        self.sigma = tf.Variable(rng.normal(shape=[self.num_features, 1]))

    #__call__ to make y hat
    def __call__(self, x):
        #phi = np.exp((-(x-self.mu)**2)/(self.sigma**2))
        #print("phi",phi)
        return tf.squeeze(tf.math.exp((-(x-self.mu)**2)/(self.sigma**2) @ self.w + self.b))

    @property
    def model(self):
        return LinearModel(
            self.w.numpy().reshape([self.num_features]), self.b.numpy().squeeze()
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

    data_generating_model = LinearModel(
        #this bias needs to be different
        weights=np_rng.integers(low=0, high=5, size=(FLAGS.num_features)), bias=2
    )
    logging.debug(data_generating_model)

    data = Data(
        data_generating_model,
        np_rng,
        FLAGS.num_features,
        FLAGS.num_samples,
        FLAGS.sigma_noise,
    )

    model = Model(tf_rng, FLAGS.num_features)
    logging.debug(model.model)

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

    logging.debug(model.model)

    # print out true values versus estimates
    print("w,    w_hat")
    compare_linear_models(data.model, model.model)


#PLOTTING

    if FLAGS.num_features > 1:
        # Only continue to plotting if x is a scalar
        exit(0)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=200)

    ax.set_title("Linear fit w clean y")
    ax.set_xlabel("x")
    ax.set_ylim(np.amax(data.clean_y) * -1.5, np.amax(data.clean_y) * 1.5)
    h = ax.set_ylabel("clean_y", labelpad=10)
    h.set_rotation(0)

    xs = np.linspace(0, 1, 10)
    xs = xs[:, np.newaxis]
    ax.plot(xs, np.squeeze(model(xs)), "-", np.squeeze(data.x), data.clean_y, "o")

    plt.tight_layout()
    plt.savefig(f"{script_path}/sincleanfit.pdf")


    fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=200)

    ax.set_title("Linear fit")
    ax.set_xlabel("x")
    ax.set_ylim(np.amax(data.clean_y) * -1.5, np.amax(data.clean_y) * 1.5)
    h = ax.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    xs = np.linspace(0, 1, 10)
    xs = xs[:, np.newaxis]
    ax.plot(xs, np.squeeze(model(xs)), "-", np.squeeze(data.x), data.y, "o")

    plt.tight_layout()
    plt.savefig(f"{script_path}/sinfit.pdf")


if __name__ == "__main__":
    app.run(main)
