from misc.utils import dense
from network.registry import register_model

_APPROX = "kron"


@register_model("mlp")
def mlp(inputs, training, config, layer_collection=None):
    for l in range(config.n_layer):
        pre, act, param = dense(inputs, output_size=config.hidden_units, is_training=training,
                                batch_norm=config.batch_norm, use_bias=config.use_bias, name="fc_"+str(l))

        if layer_collection is not None:
            layer_collection.register_fully_connected(param, inputs, pre, approx=_APPROX)
        inputs = act

    logits, _, param = dense(inputs, output_size=config.output_dim, is_training=training,
                             use_bias=config.use_bias, name="fc_"+str(config.n_layer))

    if layer_collection is not None:
        layer_collection.register_fully_connected(param, inputs, logits, approx=_APPROX)
        if config.use_fisher:
            layer_collection.register_categorical_predictive_distribution(logits, name="logit")
        else:
            layer_collection.register_normal_predictive_distribution(logits, name="mean")

    return logits, inputs
