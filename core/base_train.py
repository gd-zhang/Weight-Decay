import tensorflow as tf
from misc.summarizer import Summarizer


class BaseTrain:
    def __init__(self, sess, model, config, logger):
        self.model = model
        self.logger = logger
        if logger is not None:
            self.summarizer = Summarizer(sess, config)
        self.config = config
        self.sess = sess
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def test_epoch(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
