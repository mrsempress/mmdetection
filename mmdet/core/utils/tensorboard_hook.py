import os.path as osp
import torch

from mmcv.runner.hooks.logger.base import LoggerHook
from mmcv.runner import master_only

from .log_buffer import LogBuffer
from .summary import get_summary, get_global_step, get_writer


class TensorboardHook(LoggerHook):

    def __init__(self,
                 interval,
                 ignore_last=True,
                 reset_flag=True):
        super(TensorboardHook, self).__init__(interval, ignore_last,
                                              reset_flag)
        self.log_buffer = LogBuffer()

    @master_only
    def before_run(self, runner):
        self.writer = get_writer()

    @master_only
    def log(self, runner):
        for var in runner.log_buffer.output:
            if var in ['time', 'data_time']:
                continue
            tag = '{}/{}'.format(var, runner.mode)
            self.writer.add_scalar(tag, runner.log_buffer.output[var], get_global_step())
        for var in self.log_buffer.output:
            tag = '{}/{}'.format(var, runner.mode)
            self.writer.add_scalar(tag, self.log_buffer.output[var], get_global_step())

    @master_only
    def update_log_buffer(self, var, smooth=True):
        """

        :param var: dict, watch list var
        :param smooth: bool, if smooth, will use log buffer's update, else
        add var in output directly
        :return: None
        """
        if smooth:
            self.log_buffer.update(var)
        else:
            self.log_buffer.update_output(var)

    @master_only
    def after_train_iter(self, runner):
        self.update_log_buffer(get_summary())
        self.update_log_buffer(dict(lr=runner.current_lr()[0]), smooth=False)

        if self.every_n_inner_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
            self.log_buffer.average(self.interval)
        elif self.end_of_epoch(runner) and not self.ignore_last:
            # not precise but more stable
            runner.log_buffer.average(self.interval)
            self.log_buffer.average(self.interval)

        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()
                self.log_buffer.clear_output()

    @master_only
    def after_run(self, runner):
        self.writer.close()
