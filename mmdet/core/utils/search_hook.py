import os
import pickle
import math

from mmcv.runner.hooks import Hook
from mmcv.runner import master_only


class SearchHook(Hook):

    def __init__(self,
                 tune_epoch_start=4,
                 tune_epoch_freeze=8,
                 tune_epoch_end=8,
                 tau_start=10.,
                 tau_end=1.,
                 sin_tau=False,
                 **kwargs):
        self.tune_epoch_start = tune_epoch_start
        self.tune_epoch_freeze = tune_epoch_freeze
        self.tune_epoch_end = tune_epoch_end
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.sin_tau = sin_tau
        self.tau = self.tau_start

    def after_val_epoch(self, runner):
        if runner.mode != 'train':
            return
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        self.save_results(runner, model)

    def after_train_iter(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        epoch = runner.epoch + 1
        if self.tune_epoch_start <= epoch <= self.tune_epoch_freeze:
            t = (self.tune_epoch_freeze - self.tune_epoch_start + 1) * runner.max_inner_iter
            x = (epoch - self.tune_epoch_start) * runner.max_inner_iter + \
                runner.inner_iter
            if self.sin_tau:
                self.tau = (self.tau_start - self.tau_end) * math.sin(math.pi / 2 / t * x +
                                                                      math.pi) + self.tau_start
            else:
                self.tau = (self.tau_end - self.tau_start) / t * x + self.tau_start
        else:
            self.tau = self.tau_end
        model.reset_tau(self.tau)
        model.reset_gumbel_buffer()

    @master_only
    def save_results(self, runner, model):
        g = model.genotype()
        filepath = os.path.join(runner.work_dir, 'search_graph_' + str(runner.epoch + 1))
        try:
            model.plot(g, filepath)
        except:
            pass
        pickle.dump(g, open(os.path.join(runner.work_dir, 'search.pkl'), 'wb'))
        runner.logger.info("Searched genotype: ", str(g))
