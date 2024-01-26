from math import pi
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf

from pc_layer import add_principal_component, full_embedding
from tools_rotation import affine_transform, get_rotation_matrix


class Autoencoder(tf.keras.Model):
    def __init__(self, input_shape, bottleneck_dim):
        super().__init__()
        expected_size = (input_shape[0] // 4, input_shape[1] // 4, 8)
        self.encoder = get_encoder(input_shape, bottleneck_dim)
        self.decoder = get_decoder(expected_size, bottleneck_dim)

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


class VAE(tf.keras.Model):
    def __init__(self, input_shape, bottleneck_dim):
        super().__init__()
        expected_size = (input_shape[0] // 4, input_shape[1] // 4, 8)
        self.bottleneck_dim = bottleneck_dim
        self.encoder = get_encoder(input_shape, 2 * bottleneck_dim)
        self.decoder = get_decoder(expected_size, bottleneck_dim)

    def reparameterize(self, mean, log_var):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon

    def call(self, inputs):
        mean_log_var = self.encoder(inputs)
        mean = mean_log_var[:, :self.bottleneck_dim]
        log_var = mean_log_var[:, self.bottleneck_dim:]

        z = self.reparameterize(mean, log_var)

        reconstructed = self.decoder(z)
        return reconstructed


def get_encoder(input_shape, bottleneck_dim) -> tf.keras.Model:
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(bottleneck_dim)
    ])

    return encoder


def get_decoder(expected_size, bottleneck_dim) -> tf.keras.Model:
    decoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(bottleneck_dim,)),
        tf.keras.layers.Dense(tf.reduce_prod(expected_size), activation='relu'),
        tf.keras.layers.Reshape(expected_size),
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(1, (3, 3), padding='same')
    ])

    return decoder


def get_model(model_type: str,
              input_shape: tuple,
              n_components: int) -> tf.keras.Model:
    """
    Function to initialize projection models (except for GT-PCA).

    :param model_type: Type of model in {'Autoencoder', 'VAE'}.
    :param input_shape: Tuple specifying shape of input data.
    :param n_components: Number of components to use.
    :return: TensorFlow model.
    """
    input_shape = input_shape + (1,)
    if model_type == 'Autoencoder':
        model = Autoencoder(input_shape, n_components)
    elif model_type == 'VAE':
        model = VAE(input_shape, n_components)
    else:
        raise ValueError(f"Model type {model_type} unknown.")

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    return model


def train_autoencoder(train_data: tf.data.Dataset,
                      test_sample: np.ndarray,
                      model: tf.keras.Model,
                      training_kwargs: dict,
                      shuffle_buff: int = 1024,
                      batch_size: int = 32) -> dict:
    train_data = train_data.repeat().shuffle(shuffle_buff).map(
        lambda x: (tf.expand_dims(x, -1), tf.expand_dims(x, -1))
    ).batch(batch_size)

    model.fit(train_data, **training_kwargs)

    projections = model(test_sample[..., np.newaxis])[..., 0]

    return projections


def train_pca(train_data: tf.data.Dataset,
              test_sample: np.ndarray,
              n_components: int,
              input_shape: tuple) -> dict:
    model = PCA(n_components=n_components)

    x_train = [x.flatten() for x in train_data.as_numpy_iterator()]
    test_sample = test_sample.reshape(test_sample.shape[:1] + (-1,))

    model.fit(x_train)
    pred = model.inverse_transform(model.transform(test_sample))

    projections = pred.reshape((-1,) + input_shape)

    return projections


def train_gtpca(train_data: tf.data.Dataset,
                test_sample: np.ndarray,
                training_kwargs: dict,
                shapes: dict,
                n_components: int,
                shuffle_buff: int = 1024,
                transform: str = 'shift') -> np.ndarray:
    batch_size = training_kwargs.get('batch_size', 32)
    weights_shape = shapes['weights']
    input_shape = shapes['inputs']
    pc_kwargs = {'weight_dims': weights_shape, 'transform': transform}

    train_data = train_data.repeat().shuffle(shuffle_buff)
    train_data = train_data.map(lambda x: (x, tf.zeros((1,)))).batch(batch_size)

    model = None

    for n_comp in range(1, n_components + 1):
        if model is None:
            model = add_principal_component(pc_kwargs, input_shape=input_shape)
        else:
            model = add_principal_component(pc_kwargs, orig_model=model)

        model.fit(train_data, **training_kwargs)

    proj_model = full_embedding(model)
    projections = proj_model(test_sample)

    return projections


def load_data(classes: Optional[list] = None,
              transform: str = 'shift',
              n_rotations: int = 20) -> tuple:
    """
    Load MNIST data as TensorFlow dataset (default split).

    :param classes: If specified, only samples of a class from the list are
        passed.
    :param transform: transform, either of:
        - 'shift': Inputs are shifted (cf. Figure 1 - left of the paper).
        - 'rotation': Inputs are rotated (cf. Figure 1 - right of the paper).
    :param n_rotations: If transform == 'rotation', the images are randomly
        rotated by a multiple of 2 * pi / n_rotations. Defaults to 20.
    :return: List of datasets, dictionary of shapes.
    """
    weight_shape = original_shape = (28, 28)
    input_shape = (56, 56) if transform == 'shift' else (28, 28)

    shapes = {'weights': weight_shape,
              'original': original_shape,
              'inputs': input_shape}

    train_data, test_data = tf.keras.datasets.mnist.load_data()

    if classes:
        mask = tf.reduce_any(tf.equal(train_data[1][:, None], classes), axis=1)
        train_data = tuple(tf.boolean_mask(x, mask, axis=0) for x in train_data)

        mask = tf.reduce_any(tf.equal(test_data[1][:, None], classes), axis=1)
        test_data = tuple(tf.boolean_mask(x, mask, axis=0) for x in test_data)

    train_data = tf.data.Dataset.from_tensor_slices(train_data)
    test_data = tf.data.Dataset.from_tensor_slices(test_data)

    data = train_data, test_data
    data = tuple(ds.map(lambda x, y: tf.cast(x, tf.float32)) for ds in data)

    if transform == 'shift':
        train_data, test_data = get_padded_image(
            data, size_old=original_shape, size_new=input_shape)
    elif transform == 'rotation':
        train_data, test_data = get_rotated_image(
            data, image_shape=original_shape, n_rotations=n_rotations
        )
    else:
        train_data, test_data = data

    return train_data, test_data, shapes


def get_padded_image(
        data: tuple, size_old: tuple, size_new: tuple
) -> tuple:
    x_max, y_max = size_new[0] - size_old[0] + 1, size_new[1] - size_old[1] + 1

    def map_func(x: tf.Tensor) -> tuple:
        x_start = tf.random.uniform((1,), maxval=x_max, dtype=tf.int32)
        y_start = tf.random.uniform((1,), maxval=y_max, dtype=tf.int32)

        paddings = tf.stack(
            [tf.concat([x_start, x_max - x_start - 1], 0),
             tf.concat([y_start, y_max - y_start - 1], 0)], 0
        )

        return tf.pad(x, paddings, "CONSTANT")

    return tuple(ds.map(map_func) for ds in data)


def get_rotated_image(
        data: tuple,
        image_shape: tuple,
        n_rotations: int = 1
) -> tuple:
    rotation_matrix = tf.stack(
        [get_rotation_matrix(image_shape, 2 * pi * angle / n_rotations)
         for angle in range(n_rotations)], 0
    )

    def map_func(x: tf.Tensor) -> tuple:
        rotation_index = tf.random.uniform(shape=[], maxval=n_rotations,
                                           dtype=tf.int32)
        rotation_mat = tf.gather(rotation_matrix, rotation_index)
        x_pad = tf.expand_dims(x, -1)
        x_rot = affine_transform(image=x_pad, transform=rotation_mat)[..., 0]
        return x_rot

    return tuple(ds.map(map_func) for ds in data)


def run_example(n_components: int,
                training_kwargs: Optional[dict] = None,
                transform: str = 'shift'):
    """
    Runs experiments.

    :param n_components: Number of components to use for projections.
    :param training_kwargs: Dictionary with training kwargs for projection
        model.
    :param training_kwargs: Dictionary with training kwargs.
    """
    train_data, test_data, shapes = load_data(classes=[8], transform=transform)

    input_shape = shapes['inputs']
    results = {}

    samples = np.stack(list(test_data.take(9).as_numpy_iterator()))

    results['PCA'] = train_pca(train_data, samples, n_components, input_shape)
    results['GT-PCA'] = train_gtpca(train_data, samples, training_kwargs,
                                    shapes, n_components, transform=transform)

    for model_type in ['Autoencoder', 'VAE']:
        model = get_model(model_type, input_shape, n_components)

        results[model_type] = train_autoencoder(train_data, samples, model,
                                                training_kwargs)

    fig, axes = plt.subplots(5, 9)

    axes[0, 0].set_ylabel("Original")
    for idx, sample in enumerate(samples):
        axes[0, idx].imshow(sample, cmap='gray')

    for model_idx, (model_type, projection) in enumerate(results.items()):
        axes[model_idx + 1, 0].set_ylabel(model_type)
        for idx, proj in enumerate(projection):
            axes[model_idx + 1, idx].imshow(proj, cmap='gray')

    for ax in axes.flat:
        ax.set_xticks([], [])
        ax.set_yticks([], [])

    plt.show()


if __name__ == '__main__':
    # training_kwargs = {'verbose': 1, 'epochs': 1, 'steps_per_epoch': 2500}
    # n_components = 10
    training_kwargs = {'verbose': 1, 'epochs': 1, 'steps_per_epoch': 1000}
    n_components = 4

    for transform in ['shift', 'rotation']:
        run_example(n_components, training_kwargs=training_kwargs,
                    transform=transform)
