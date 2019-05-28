import tensorflow as tf

from misc.utils import dense, conv2d
from network.registry import register_model

_OUTPUT = [64, 128, 256]
_APPROX = "kron"


def BasicBlock(inputs, training, output_dim, name, stride=1, batch_norm=False, use_bias=False, layer_collection=None):
    pre1, act1, param1 = conv2d(inputs, kernel_size=(3, 3), out_channels=output_dim, strides=(stride, stride),
                                is_training=training, batch_norm=batch_norm, use_bias=use_bias, name=name+"conv1")

    pre2, act2, param2 = conv2d(act1, kernel_size=(3, 3), out_channels=output_dim, activation_fn=tf.identity,
                                is_training=training, batch_norm=batch_norm, use_bias=use_bias, name=name+"conv2")

    if layer_collection is not None:
        layer_collection.register_conv2d(param1, (1, stride, stride, 1), "SAME", inputs, pre1, approx=_APPROX)
        layer_collection.register_conv2d(param2, (1, 1, 1, 1), "SAME", act1, pre2, approx=_APPROX)

    if stride != 1:
        pre3, act3, param3 = conv2d(inputs, kernel_size=(1, 1), out_channels=output_dim, strides=(stride, stride),
                                    is_training=training, batch_norm=batch_norm, use_bias=use_bias,
                                    name=name+"conv_skip", activation_fn=tf.identity)
        if layer_collection is not None:
            layer_collection.register_conv2d(param3, (1, stride, stride, 1), "SAME", inputs, pre3, approx=_APPROX)

        return tf.nn.relu(act2 + act3)

    return tf.nn.relu(act2 + inputs)


def ResNet(inputs, training, num_blocks, output_dim, use_fisher=False,
           batch_norm=False, use_bias=False, layer_collection=None):
    pre1, act1, param1 = conv2d(inputs, kernel_size=(3, 3), out_channels=64, use_bias=use_bias,
                                is_training=training, batch_norm=batch_norm, name="conv1")
    if layer_collection is not None:
        layer_collection.register_conv2d(param1, (1, 1, 1, 1), "SAME", inputs, pre1, approx=_APPROX)
    out = act1
    for i, b in enumerate(num_blocks):
        for l in range(b):
            if i > 0 and l == 0:
                stride = 2
            else:
                stride = 1
            out = BasicBlock(out, training, _OUTPUT[i], name="Res_"+str(i+1)+"Blk_"+str(l+1), use_bias=use_bias,
                             stride=stride, batch_norm=batch_norm, layer_collection=layer_collection)

    # average pooling
    assert out.get_shape().as_list()[1:] == [8, 8, 256]
    out = tf.reduce_mean(out, [1, 2])
    assert out.get_shape().as_list()[1:] == [256]

    logits, _, param = dense(out, output_size=output_dim, is_training=training, use_bias=use_bias, name="fc")
    if layer_collection is not None:
        layer_collection.register_fully_connected(param, out, logits, approx=_APPROX)
        if use_fisher:
            layer_collection.register_categorical_predictive_distribution(logits, name="logit")
        else:
            layer_collection.register_normal_predictive_distribution(logits, name="mean")
    return logits


@register_model("resnet20")
def resnet20(inputs, training, config, layer_collection=None):
    return ResNet(inputs, training, [3, 3, 3], config.output_dim, config.use_fisher,
                  config.batch_norm, config.use_bias, layer_collection)


@register_model("resnet32")
def resnet32(inputs, training, config, layer_collection=None):
    return ResNet(inputs, training, [5, 5, 5], config.output_dim, config.use_fisher,
                  config.batch_norm, config.use_bias, layer_collection)


@register_model("resnet44")
def resnet44(inputs, training, config, layer_collection=None):
    return ResNet(inputs, training, [7, 7, 7], config.output_dim, config.use_fisher,
                  config.batch_norm, config.use_bias, layer_collection)


@register_model("resnet56")
def resnet56(inputs, training, config, layer_collection=None):
    return ResNet(inputs, training, [9, 9, 9], config.output_dim, config.use_fisher,
                  config.batch_norm, config.use_bias, layer_collection)
