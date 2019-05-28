from core.base_train import BaseTrain
from tqdm import tqdm
from misc.utils import SetFromFlat, GetFlat, unflatten, flatten, numel
import numpy as np
import tensorflow as tf


class Trainer(BaseTrain):
    def __init__(self, sess, model, train_loader, test_loader, config, logger):
        super(Trainer, self).__init__(sess, model, config, logger)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.get_params = GetFlat(self.sess, self.model.params_net)
        self.set_params = SetFromFlat(self.sess, self.model.params_net)
        self.unflatten = unflatten(self.model.params_net)
        self.norm_list = []

        self.summary_op = tf.summary.merge_all()

    def init_kfac(self):
        self.logger.info('Roger Initialization!')
        self.model.optim._fisher_est.reset(self.sess)

        for itr, (x, y) in enumerate(self.train_loader):
            feed_dict = {
                self.model.inputs: x,
                # self.model.targets: y,
                self.model.is_training: True
            }
            self.sess.run(self.model.optim.init_cov_op, feed_dict=feed_dict)
        self.model.optim._fisher_est.rescale(self.sess, 1. / len(self.train_loader))

        # inverse
        if self.model.inv_update_op is not None:
            self.sess.run(self.model.inv_update_op)

        self.logger.info('Done Roger Initialization!')

    def train(self):
        if self.config.roger_init:
            self.init_kfac()
        for cur_epoch in range(self.config.epoch):
            self.logger.info('epoch: {}'.format(int(cur_epoch)))
            self.train_epoch()
            self.test_epoch()

            if cur_epoch % 100 == 0:
                self.model.save(self.sess)

    def train_epoch(self):
        loss_list = []
        acc_list = []

        for itr, (x, y) in enumerate(tqdm(self.train_loader)):
            feed_dict = {
                self.model.inputs: x,
                self.model.targets: y,
                self.model.is_training: True,
            }
            self.sess.run(self.model.train_op, feed_dict=feed_dict)
            cur_iter = self.model.global_step_tensor.eval(self.sess)

            if cur_iter % self.config.get('TCov', 10) == 0 and self.model.cov_update_op is not None:
                self.sess.run(self.model.cov_update_op, feed_dict=feed_dict)

            if cur_iter % self.config.get('TInv', 100) == 0 and self.model.inv_update_op is not None:
                self.sess.run(self.model.inv_update_op)

        for itr, (x, y) in enumerate(self.train_loader):
            feed_dict = {
                self.model.inputs: x,
                self.model.targets: y,
                self.model.is_training: True
            }

            loss, acc = self.sess.run(
                [self.model.loss, self.model.acc],
                feed_dict=feed_dict)
            loss_list.append(loss)
            acc_list.append(acc)

        avg_loss = np.mean(loss_list)
        avg_acc = np.mean(acc_list)
        self.logger.info("[Train] loss: %5.4f | accuracy: %5.4f"%(float(avg_loss), float(avg_acc)))

        l2_norm = self.sess.run(self.model.l2_norm)
        self.logger.info("l2_norm: %5.4f"%(float(l2_norm)))

        # summarize
        summaries_dict = dict()
        summaries_dict['train_loss'] = avg_loss
        summaries_dict['train_acc'] = avg_acc
        summaries_dict['l2_norm'] = l2_norm

        # summarize
        cur_iter = self.model.global_step_tensor.eval(self.sess)
        self.summarizer.summarize(cur_iter, summaries_dict=summaries_dict)

    def test_epoch(self):
        loss_list = []
        acc_list = []
        for (x, y) in self.test_loader:
            feed_dict = {
                self.model.inputs: x,
                self.model.targets: y,
                self.model.is_training: False
            }
            loss, acc = self.sess.run([self.model.loss, self.model.acc], feed_dict=feed_dict)
            loss_list.append(loss)
            acc_list.append(acc)

        avg_loss = np.mean(loss_list)
        avg_acc = np.mean(acc_list)
        self.logger.info("[Test] loss: %5.4f | accuracy: %5.4f"%(float(avg_loss), float(avg_acc)))

        # summarize
        summaries_dict = dict()
        summaries_dict['test_loss'] = avg_loss
        summaries_dict['test_acc'] = avg_acc

        # summarize
        cur_iter = self.model.global_step_tensor.eval(self.sess)
        self.summarizer.summarize(cur_iter, summaries_dict=summaries_dict)


