import re
import os

from mmcv.runner.hooks.logger.base import LoggerHook
from mmcv.runner import master_only
from mmdet.core.utils import logger


class CheckpointHook(LoggerHook):

    def __init__(self,
                 save_every_n_steps,
                 max_to_keep=2,
                 keep_every_n_epochs=50,
                 keep_in_n_epoch=[],
                 ignore_last=True,
                 reset_flag=True,
                 **kwargs):
        super(CheckpointHook, self).__init__(save_every_n_steps, ignore_last, reset_flag)
        self.save_every_n_steps = save_every_n_steps
        self.max_to_keep = max_to_keep
        self.keep_every_n_epochs = keep_every_n_epochs
        self.keep_in_n_epoch = keep_in_n_epoch

    def clear_extra_checkpoints(self, work_dir):
        rm_filename_dict = dict()
        for filename in os.listdir(work_dir):
            if not filename.endswith('.pth') or 'iter' not in filename:
                continue
            # keep the checkpoints which are protected.
            iter_n = re.search("epoch_[0-9]*_iter_([0-9]*).pth", filename)
            if iter_n:
                rm_filename_dict[int(iter_n.group(1))] = filename
        sort_iter_list = sorted(rm_filename_dict.keys())
        for rm_iter in sort_iter_list[: -self.max_to_keep]:
            os.remove(os.path.join(work_dir, rm_filename_dict[rm_iter]))

    def save_checkpoint(self, runner, protect=False, offset=1, iter_no_offset=False):
        epoch_n, iter_n = runner.epoch + offset, runner.iter + offset
        if iter_no_offset:
            iter_n = runner.iter
        filename_tmpl = 'epoch_{}.pth'
        if not protect:
            filename_tmpl = 'epoch_{{0}}_iter_{}.pth'.format(iter_n)

        runner.save_checkpoint(runner.work_dir,
                               filename_tmpl=filename_tmpl,
                               save_optimizer=True,
                               offset=offset,
                               iter_no_offset=iter_no_offset)
        logger.info("The new checkpoint has been saved to [{}]".format(os.path.join(
            runner.work_dir, filename_tmpl.format(epoch_n))))
        try:
            self.clear_extra_checkpoints(runner.work_dir)
        except:
            logger.warn("Cannot remove the old checkpoints.")

    @master_only
    def after_train_iter(self, runner):
        # note that when after_train_iter is called, the iter num has not plus 1 yet.
        if self.every_n_iters(runner, self.save_every_n_steps):
            self.save_checkpoint(runner)

    # @master_only
    # def before_train_epoch(self, runner):
    #     if runner.epoch > 0:
    #         self.save_checkpoint(runner, protect=False, iter_no_offset=False)

    @master_only
    def after_train_epoch(self, runner):
        # note that when after_train_iter is called, the epoch num has not plus 1 yet.
        if self.every_n_epochs(runner, self.keep_every_n_epochs) or \
                (runner.epoch + 1) in self.keep_in_n_epoch:
            self.save_checkpoint(runner, protect=True, iter_no_offset=True)
        else:
            self.save_checkpoint(runner, protect=False, iter_no_offset=True)

    @master_only
    def after_run(self, runner):
        # note that when after_run is called, the iter and epoch num has already plus 1.
        self.save_checkpoint(runner, offset=0)
