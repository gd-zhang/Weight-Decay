import tensorflow as tf
import numpy as np

from misc.utils import dense, conv2d
from network.registry import register_model

_OUTPUT = [64, 128, 256, 512, 512]
_APPROX = "kron"


def VGG(inputs, layer, training, config, layer_collection=None):
    for b in range(5):
        for l in range(layer[b]):
            pre, act, param = conv2d(inputs, kernel_size=(3, 3),
                                     out_channels=_OUTPUT[b],
                                     is_training=training,
                                     batch_norm=config.batch_norm,
                                     use_bias=config.use_bias,
                                     name="conv_"+str(b)+"_"+str(l))
            if layer_collection is not None:
                layer_collection.register_conv2d(param, (1, 1, 1, 1), "SAME", inputs, pre, approx=_APPROX)
            inputs = act

        inputs = tf.layers.max_pooling2d(inputs, 2, 2, "SAME")

    flat = tf.reshape(inputs, shape=[-1, int(np.prod(inputs.shape[1:]))])
    logits, _, param = dense(flat, output_size=config.output_dim, use_bias=config.use_bias,
                             is_training=training, name="fc")

    if layer_collection is not None:
        layer_collection.register_fully_connected(param, flat, logits, approx=_APPROX)
        if config.use_fisher:
            layer_collection.register_categorical_predictive_distribution(logits, name="logit")
        else:
            layer_collection.register_normal_predictive_distribution(logits, name="mean")

    return logits


@register_model("vgg11")
def vgg11(inputs, training, config, layer_collection=None):
    return VGG(inputs, [1, 1, 2, 2, 2], training, config, layer_collection)


@register_model("vgg13")
def vgg13(inputs, training, config, layer_collection=None):
    return VGG(inputs, [2, 2, 2, 2, 2], training, config, layer_collection)


@register_model("vgg16")
def vgg16(inputs, training, config, layer_collection=None):
    return VGG(inputs, [2, 2, 3, 3, 3], training, config, layer_collection)


@register_model("vgg19")
def vgg19(inputs, training, config, layer_collection=None):
    return VGG(inputs, [2, 2, 4, 4, 4], training, config, layer_collection)
