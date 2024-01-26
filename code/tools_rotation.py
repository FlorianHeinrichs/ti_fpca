import tensorflow as tf


def affine_transform(
    image,
    transform,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0.0,
    data_format="channels_last",
):
    initial_shape = image.shape
    if len(initial_shape) == 2:
        image = tf.reshape(image, (1,) + initial_shape + (1,))
    if len(initial_shape) == 3:
        image = tf.expand_dims(image, axis=0)

    if len(initial_shape) not in (2, 3, 4):
        raise ValueError(
            "Invalid image rank: expected rank 2 (single gray scale image),"
            " rank 3 (single image) or rank 4 (batch of images). Received input"
            f" with shape: image.shape={initial_shape}"
        )

    if len(transform.shape) == 1:
        transform = tf.expand_dims(transform, axis=0)
    elif len(transform.shape) not in (1, 2):
        raise ValueError(
            "Invalid transform rank: expected rank 1 (single transform) "
            "or rank 2 (batch of transforms). Received input with shape: "
            f"transform.shape={transform.shape}"
        )

    if data_format == "channels_first":
        image = tf.transpose(image, (0, 2, 3, 1))

    affined = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=tf.cast(transform, dtype=tf.float32),
        output_shape=tf.shape(image)[1:-1],
        fill_value=fill_value,
        interpolation=interpolation.upper(),
        fill_mode=fill_mode.upper(),
    )
    affined = tf.ensure_shape(affined, image.shape)

    if data_format == "channels_first":
        affined = tf.transpose(affined, (0, 3, 1, 2))

    if len(initial_shape) == 2:
        affined = affined[0, ..., 0]
    elif len(initial_shape) == 3:
        affined = affined[0, ...]

    return affined


def get_rotation_matrix(shape: tuple, angle: float):
    if len(shape) == 4:
        _, image_height, image_width, _ = shape
        image_height = shape[1]
        image_width = shape[2]
    elif len(shape) == 3:
        image_height, image_width, _ = shape
    else:
        image_height, image_width = shape

    cos_theta = tf.math.cos(angle)
    sin_theta = tf.math.sin(angle)
    image_height = tf.cast(image_height, tf.float32)
    image_width = tf.cast(image_width, tf.float32)

    x_offset = (
        (1 - cos_theta) * (image_width - 1) + sin_theta * (image_height - 1)
    ) / 2.0
    y_offset = (
        (1 - cos_theta) * (image_height - 1) - sin_theta * (image_width - 1)
    ) / 2.0

    outputs = tf.stack([cos_theta, -sin_theta, x_offset,
                        sin_theta,  cos_theta, y_offset, 0, 0], axis=0)
    outputs = tf.expand_dims(outputs, 0)
    if len(shape) == 3:
        outputs = tf.squeeze(outputs, axis=0)

    return outputs
