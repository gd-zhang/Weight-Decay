import tensorflow as tf

from libs.kfac import optimizer as kfac_opt
from libs.kfac import layer_collection as lc
from libs.sgd import optimizer as sgd_opt
from libs.adam import optimizer as adam_opt
from misc.utils import flatten, unflatten
from network.registry import get_model
from core.base_model import BaseModel


class Model(BaseModel):
    def __init__(self, config, sess):
        super().__init__(config)
        self.sess = sess
        self.build_model()
        self.init_optim()
        self.init_saver()

    @property
    def params_net(self):
        return tf.trainable_variables('network')

    @property
    def params_all(self):
        return tf.global_variables()

    @property
    def params_w_flatten(self):
        return flatten(self.params_net)

    @property
    def params_w_flatten_last(self):
        return flatten(self.params_net[-2:])

    def init_saver(self):
        self.saver = tf.train.Saver(var_list=self.params_all,
                                    max_to_keep=self.config.max_to_keep)

    def build_model(self):
        self.inputs = tf.placeholder(tf.float32, [None] + self.config.input_dim)
        self.targets = tf.placeholder(tf.int64, [None])
        self.is_training = tf.placeholder(tf.bool)
        if self.config.use_kfac:
            self.layer_collection = lc.LayerCollection()
        else:
            self.layer_collection = None
            self.cov_update_op = None
            self.inv_update_op = None

        inputs = self.inputs
        with tf.variable_scope("network"):
            net = get_model(self.config.model_name)
            logits = net(inputs, self.is_training, self.config, self.layer_collection)

        self.acc = tf.reduce_mean(tf.cast(tf.equal(
            self.targets, tf.argmax(logits, axis=1)), dtype=tf.float32))

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.targets, logits=logits))

        self.l2_norm = tf.reduce_sum(tf.square(self.params_w_flatten))

    def init_optim(self):
        if self.config.optimizer == "sgd":
            self.optim = sgd_opt.SGDOptimizer(
                tf.train.exponential_decay(self.config.learning_rate,
                                           self.global_step_tensor,
                                           self.config.decay_every_itr, 0.1, staircase=True),
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                weight_decay_type=self.config.weight_decay_type,
                weight_list=self.config.weight_list)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optim.minimize(self.loss, global_step=self.global_step_tensor)
        elif self.config.optimizer == "adam":
            self.optim = adam_opt.ADAMOptimizer(
                tf.train.exponential_decay(self.config.learning_rate,
                                           self.global_step_tensor,
                                           self.config.decay_every_itr, 0.1, staircase=True),
                weight_decay=self.config.weight_decay,
                weight_decay_type=self.config.weight_decay_type,
                weight_list=self.config.weight_list)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optim.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            kl_clip = self.config.get("kl_clip", None)
            self.optim = kfac_opt.KFACOptimizer(
                tf.train.exponential_decay(self.config.learning_rate,
                                           self.global_step_tensor,
                                           self.config.decay_every_itr, 0.1, staircase=True),
                cov_ema_decay=self.config.cov_ema_decay,
                damping=self.config.damping,
                layer_collection=self.layer_collection,
                norm_constraint=kl_clip,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                weight_decay_type=self.config.weight_decay_type,
                weight_list=self.config.weight_list)

            self.cov_update_op = self.optim.cov_update_op
            self.inv_update_op = self.optim.inv_update_op

            self.update_ops = update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optim.minimize(self.loss, global_step=self.global_step_tensor)
