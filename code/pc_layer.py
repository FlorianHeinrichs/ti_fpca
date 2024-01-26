from math import pi
from typing import Optional, Tuple, Union

import tensorflow as tf

from custom_metrics import NegativeMAE, NegativeMSE
from tools_rotation import affine_transform, get_rotation_matrix


class GTPCA_Layer(tf.keras.layers.Layer):
    """
    TensorFlow layer to approximate general transform-invariant principal
    component (GT-PC).
    """

    def __init__(self, weight_dims: Tuple, transform: str = 'shift', **kwargs):
        """
        Initialize layer.

        :param weight_dims: Dimension of weights, e.g.:
            - (28, 28) for MNIST (grayscale)
            - (28, 28, 3) for MNIST (RGB)
            - (1400, 2) for "Handwriting" dataset
        :param transform: Transform to apply (either 'shift' or 'rotation').
            Defaults to 'shift'.
        """
        super().__init__(**kwargs)
        self.transform = transform

        self.weight_dims = weight_dims
        self.n_dims = len(weight_dims)
        self.axis = list(range(1 + self.n_dims, 2 * self.n_dims + 1))
        self.w = self.add_weight(
            shape=self.weight_dims, trainable=True, name='weights',
        )
        self.n_rotations = kwargs.get('n_rotations', 20)

        self.input_dims = None
        self.n = None
        self.get_filter = None
        self.call_func = None
        self.max_argmax = None
        self.project = False
        self.get_scores = False
        self.get_argmax = False

    def build(self, input_shape):
        if len(input_shape) - 1 != self.n_dims:
            msg = ("Number of input dimensions does not match number of weight"
                   f" dimensions: {len(input_shape)} - 1 != {self.n_dims}")
            raise ValueError(msg)

        dims_in, dims_w = input_shape[1:], self.weight_dims

        self.input_dims = dims_in
        self.n = tf.cast(tf.reduce_prod(dims_in), dtype=tf.float32)
        self.max_argmax = tf.constant(
            [[abs(d1 - d2) + 1 for d1, d2 in zip(dims_in, dims_w)]])

        self.call_func = self.call_main

        if dims_in == dims_w:
            self.axis = list(range(2, self.n_dims + 2))
            self.n_dims = 1

            if self.transform in [None, 'shift']:
                self.transform = 'rotation'
                self.n_rotations = 1

            if self.transform == 'rotation':
                self.max_argmax = self.n_rotations

        elif self.transform == 'shift':
            if all(d1 >= d2 for d1, d2 in zip(dims_in, dims_w)):
                self.call_func = self.call_input_larger
            elif all(d1 <= d2 for d1, d2 in zip(dims_in, dims_w)):
                self.n = tf.cast(tf.reduce_prod(dims_w), dtype=tf.float32)
            else:
                raise ValueError("Shapes of input and weights incompatible.")
        else:
            raise ValueError("Shapes of input and weights are not compatible.")

    def call(self, inputs, training: bool = False):
        return self.call_func(inputs, training=training)

    def call_input_larger(self, inputs, training: bool = False):
        """
        Efficient implementation of the call method when inputs are larger than
        weights.
        """
        weights = self.w / tf.math.sqrt(tf.reduce_sum(self.w ** 2) / self.n)
        scores = tf.nn.convolution(inputs[..., None],
                                   weights[..., None, None])[..., 0] / self.n

        axis = range(1, self.n_dims + 1)

        if not self.project:
            output = tf.reduce_max(tf.abs(scores), axis=axis)
            output = tf.expand_dims(output, -1)
        else:
            argmax = get_argmax(tf.abs(scores), self.n_dims)

            argmax_pad = tf.stack([argmax, self.max_argmax - argmax - 1], -1)
            weights_padded = tf.map_fn(
                lambda x: tf.pad(weights, x), argmax_pad,
                fn_output_signature=tf.TensorSpec(self.input_dims)
            )

            scores = tf.gather_nd(scores, argmax, batch_dims=1)
            scores = tf.reshape(scores, [-1] + [1] * self.n_dims)

            output = scores * weights_padded

            if self.get_scores:
                max_scores = tf.reduce_max(tf.abs(scores), axis=axis)
                max_scores = tf.expand_dims(max_scores, -1)

                if self.get_argmax:
                    output = output, max_scores, argmax / self.max_argmax
                else:
                    output = output, max_scores

        return output

    def call_main(self, inputs, training: bool = False):
        weights = self._calc_transforms()
        weights = tf.expand_dims(weights, axis=0)
        weights_norm = tf.sqrt(tf.reduce_sum(
            tf.square(weights), axis=self.axis, keepdims=True
        ) / self.n)
        weights_normed = weights / weights_norm

        input_shape = [-1] + [1] * self.n_dims + list(self.input_dims)
        inputs_exp = tf.reshape(inputs, input_shape)
        scores = tf.reduce_sum(
            inputs_exp * weights_normed, axis=self.axis
        ) / self.n

        axis = range(1, self.n_dims + 1)

        if not self.project:
            output = tf.reduce_max(tf.abs(scores), axis=axis)
            output = tf.expand_dims(output, -1)
        else:
            argmax = get_argmax(tf.abs(scores), self.n_dims)

            batch_size = tf.shape(argmax)[0]
            w_shape = [batch_size] + (self.n_dims + len(self.weight_dims)) * [1]
            weights_normed = tf.tile(weights_normed, w_shape)
            weights_normed = tf.gather_nd(weights_normed, argmax, batch_dims=1)
            scores_max = tf.gather_nd(scores, argmax, batch_dims=1)
            w_shape = [batch_size] + len(self.weight_dims) * [1]
            output = tf.reshape(scores_max, w_shape) * weights_normed

            if self.get_scores:
                max_scores = tf.reduce_max(tf.abs(scores), axis=axis)
                max_scores = tf.expand_dims(max_scores, -1)

                if self.get_argmax:
                    output = output, max_scores, argmax / self.max_argmax
                else:
                    output = output, max_scores

        return output

    def _calc_transforms(self) -> tf.Tensor:
        """
        Calculate all valid transforms of current weights self.w with
        dimensions self.weight_dims.

        :return: Tensor with transformed weights of size:
            - (n_valid_transforms,) + self.weight_dims
        """
        if self.transform == 'shift':
            return self._calc_shift()
        elif self.transform == 'rotation':
            return self._calc_rotation()
        elif self.transform is None:
            return self.w
        else:
            raise ValueError(f"Transform {self.transform} unknown.")

    def _calc_shift(self) -> tf.Tensor:
        if self.n_dims > 3:
            raise ValueError(f"Shift not implemented for {self.n_dims} > 3.")

        new_shape = (1,) * (4 - len(self.weight_dims)) + (*self.weight_dims, 1)
        ksizes = [1] * (4 - len(self.input_dims)) + [*self.input_dims, 1]
        indices = [0] * (4 - len(self.weight_dims)) + [...]
        weights = tf.extract_volume_patches(
            tf.reshape(self.w, new_shape), ksizes, [1] * 5, "VALID"
        )[indices]
        weight_shape = tuple(
            w - i + 1 for w, i in zip(self.weight_dims, self.input_dims)
        ) + self.input_dims
        weights = tf.reshape(weights, weight_shape)

        return weights

    def _calc_rotation(self) -> tf.Tensor:
        n_rot = self.n_rotations
        rotation_matrix = tf.concat(
            [get_rotation_matrix(self.weight_dims, 2 * pi * angle / n_rot)
             for angle in range(n_rot)], 0
        )
        weights = tf.expand_dims(tf.stack([self.w] * n_rot, 0), -1)
        weights = affine_transform(image=weights, transform=rotation_matrix)

        return weights[..., 0]


def add_principal_component(
        pc_kwargs: dict,
        input_shape: Optional[tuple] = None,
        orig_model: Optional[tf.keras.Model] = None,
        loss: Union[str, tf.keras.losses.Loss] = 'NegativeMSE'
) -> tf.keras.Model:
    """
    Add layer to estimate next GT-PC to model. If no layer exists yet, create
    a new model. Only the last layer/component is trainable.

    :param pc_kwargs: Keyword arguments for construction of GT-PC layer.
    :param input_shape: Shape of input data.
    :param orig_model: Model containing GT-PC layers. Defaults to None (no layer
        exists).
    :param loss: Loss function used during model training. Only supports
        'NegativeMSE' and 'NegativeMAE'.
    :return: Model to estimate last component.
    """
    if loss == 'NegativeMSE':
        loss = NegativeMSE()
    elif loss == 'NegativeMAE':
        loss = NegativeMAE()
    elif isinstance(loss, str):
        raise ValueError(f"Loss {loss} unknown.")

    if pc_kwargs is None:
        pc_kwargs = {}

    if orig_model is None and input_shape is not None:
        inputs = tf.keras.layers.Input(shape=input_shape)
        pc_layer = GTPCA_Layer(**pc_kwargs)(inputs)
    elif orig_model is not None:
        input_shape = orig_model.layers[0].input_shape[0][1:]
        inputs = tf.keras.layers.Input(shape=input_shape)
        proj_input = proj_output = inputs
        for layer in orig_model.layers[1:]:
            layer.trainable = False
            layer.project = True
            layer.get_scores = False

            if 'subtract' in layer.name:
                proj_input = layer([proj_input, proj_output])
            else:
                proj_output = layer(proj_input)

        difference = tf.keras.layers.subtract([proj_input, proj_output])

        pc_layer = GTPCA_Layer(**pc_kwargs)(difference)
    else:
        raise ValueError("Either an input shape or an existing model needs to "
                         "be provided.")

    model = tf.keras.Model(inputs=inputs, outputs=pc_layer)
    model.compile(loss=loss, optimizer='adam', metrics=['mae', 'mse'])

    return model


def full_embedding(orig_model: tf.keras.Model,
                   get_scores: bool = False,
                   get_argmax: bool = False) -> tf.keras.Model:
    """
    Takes a model containing GT-PC layers and constructs a model to:
        - return reconstructions (in original space)
        - return projections (in low dimensional space)
        - return projections and argmax (in low dimensional space)

    :param orig_model: Original TensorFlow model containing GT-PC layers.
    :param get_scores: If True, return projections; else reconstructions.
    :param get_argmax: If True, return argmax in addition to projections.
    :return: TensorFlow model containing GT-PC layers.
    """
    input_shape = orig_model.layers[0].input_shape[0][1:]
    inputs = tf.keras.layers.Input(shape=input_shape)

    projections, scores, argmax = [], [], []
    proj_input = proj_output = inputs
    for layer in orig_model.layers[1:]:
        layer.trainable = False
        layer.project = True
        layer.get_scores = get_scores
        layer.get_argmax = get_argmax

        if 'subtract' in layer.name:
            proj_input = layer([proj_input, proj_output])
        else:
            proj_output = layer(proj_input)

        if isinstance(proj_output, tuple):
            if len(proj_output) == 2:
                proj_output, score = proj_output
                scores.append(score)
            elif len(proj_output) == 3:
                proj_output, score, arg_m = proj_output
                scores.append(score)
                argmax.append(arg_m)

        if isinstance(layer, GTPCA_Layer):
            projections.append(proj_output)

    if get_scores and get_argmax:
        output = tf.keras.layers.Concatenate()(scores + argmax)
    elif get_scores:
        output = tf.keras.layers.Concatenate()(scores)
    else:
        output = tf.keras.layers.Add()(projections)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'])

    return model


def get_argmax(
        inputs: tf.Tensor, n_dims: int
) -> tf.Tensor:
    """
    Auxiliary function to get the argmax of a tensor of arbitrary dimension. The
    argmax is calculated along the last n dimensions (as specified by n_dims).
    If multiple argmax exist, only the first occurrence is used.

    :param inputs: Tensor whose argmax to find.
    :param n_dims: Number of axes along which to calculate the argmax.
    :return: Tensor with coordinates of size (batch, n_dims).
    """
    indices = tf.math.equal(
        inputs, tf.reduce_max(inputs, keepdims=True, axis=range(1, n_dims + 1))
    )
    indices_as_ints = tf.cast(indices, dtype=tf.int32)

    new_shape = (-1, tf.reduce_prod(tf.shape(inputs)[1:]))
    orig_shape = tf.concat([tf.constant([-1]), tf.shape(inputs)[1:]], 0)
    indices_cumsum = tf.math.cumsum(tf.reshape(indices_as_ints, new_shape),
                                    axis=-1)
    indices_cumsum = tf.reshape(indices_cumsum, orig_shape)

    indices = tf.math.logical_and(indices, indices_cumsum <= 1)
    argmax = tf.where(indices)[:, 1:]
    argmax = tf.cast(argmax, dtype=tf.int32)
    return argmax
