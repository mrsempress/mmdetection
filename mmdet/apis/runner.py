import logging
import os.path as osp
import time
import numpy as np
import pickle
from tqdm import tqdm
from itertools import islice

import mmcv
import torch

from mmcv.runner import hooks
from mmcv.runner.hooks import (Hook, LrUpdaterHook, CheckpointHook, IterTimerHook,
                               OptimizerHook, lr_updater)
from mmcv.runner.checkpoint import load_checkpoint, save_checkpoint
from mmcv.runner.priority import get_priority
from mmcv.runner import get_dist_info, get_host_info, get_time_str, obj_from_dict

from mmdet.core.utils.log_buffer import LogBuffer
from mmdet.core.utils.summary import (
    get_summary, reset_global_step, get_global_step, set_epoch, set_total_epoch, set_inner_iter,
    set_total_inner_iter, update_global_step)


class Runner(object):
    """A training helper for PyTorch.

    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        batch_processor (callable): A callable method that process a data
            batch. The interface of this method should be
            `batch_processor(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`torch.optim.Optimizer`): If it is a dict,
            runner will construct an optimizer according to it.
        work_dir (str, optional): The working directory to save checkpoints
            and logs.
        log_level (int): Logging level.
        logger (:obj:`logging.Logger`): Custom logger. If `None`, use the
            default logger.
    """

    def __init__(self,
                 model,
                 batch_processor,
                 optimizer=None,
                 search_optimizer=None,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger=None,
                 runner_attr_dict=None):
        assert callable(batch_processor)
        assert logger, "Please provide the logger."
        self.model = model
        if optimizer is not None:
            self.optimizer = self.init_optimizer(optimizer)
        else:
            self.optimizer = None
        if search_optimizer is not None:
            self.search_optimizer = self.init_optimizer(search_optimizer, 'arch_parameters')
            self.search_step = False
        else:
            self.search_optimizer = None
        self.batch_processor = batch_processor

        # create work_dir
        if mmcv.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            mmcv.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        if runner_attr_dict:
            # set attributes to runner
            for k, v in runner_attr_dict.items():
                setattr(self, k, v)

        self._rank, self._world_size = get_dist_info()
        self._checkpoint_batchsize = None
        self._checkpoint_global_step = None
        self._checkpoint_inner_iter = None
        self.batchsize = self._world_size * getattr(self, 'imgs_per_gpu', 1)
        self.timestamp = get_time_str()
        self.logger = logger
        self.log_buffer = LogBuffer()

        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0  # store the actual iter even batch size changed
        self._inner_iter = 0
        self._max_inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_inner_iter(self):
        """int: Maximum iteration in an epoch."""
        return self._max_inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    def init_optimizer(self, optimizer, param_attr='parameters'):
        """Init the optimizer.

        Args:
            optimizer (dict or :obj:`~torch.optim.Optimizer`): Either an
                optimizer object or a dict used for constructing the optimizer.

        Returns:
            :obj:`~torch.optim.Optimizer`: An optimizer object.

        Examples:
            >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
            >>> type(runner.init_optimizer(optimizer))
            <class 'torch.optim.sgd.SGD'>
        """
        if hasattr(self.model, 'module'):
            model = self.model.module
        if isinstance(optimizer, dict):
            optimizer = obj_from_dict(
                optimizer, torch.optim, dict(params=getattr(model, param_attr)()))
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                'optimizer must be either an Optimizer object or a dict, '
                'but got {}'.format(type(optimizer)))
        return optimizer

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list: Current learning rate of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return [group['lr'] for group in self.optimizer.param_groups]

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def build_hook(self, args, hook_type=None):
        if isinstance(args, Hook):
            return args
        elif isinstance(args, dict):
            assert issubclass(hook_type, Hook)
            return hook_type(**args)
        else:
            raise TypeError('"args" must be either a Hook object'
                            ' or dict, not {}'.format(type(args)))

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               self.logger)

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=dict(),
                        offset=1,
                        iter_no_offset=False):
        global_step = get_global_step()
        # when we save checkpoint after the epoch finished, the iter has already plus 1.
        iter_offset = 0 if iter_no_offset else offset
        meta.update(epoch=self.epoch + offset, iter=self.iter + iter_offset,
                    inner_iter=self.inner_iter + iter_offset,
                    global_step=global_step, batchsize=self.batchsize,
                    initial_lr=self.initial_lr)

        filename = osp.join(out_dir, filename_tmpl.format(self.epoch + offset))
        local_filename = "./{}".format(filename_tmpl.format(self.epoch + offset))
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filename, optimizer=optimizer, meta=meta)
        mmcv.symlink(local_filename, osp.join(out_dir, 'latest.pth'))

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        if isinstance(data_loader, list):
            assert len(data_loader) == 2 and self.search_optimizer is not None
            self.search_data_loader = data_loader[1]
            data_loader = data_loader[0]

        self.data_loader = data_loader
        max_global_step = self._max_epochs * len(data_loader) * self.batchsize
        global_step_left = max_global_step - get_global_step()
        self._max_iters = self.iter + global_step_left // self.batchsize

        inner_iter_left = len(data_loader)
        if self._checkpoint_batchsize and self._checkpoint_inner_iter:
            checkpoint_data_loader_len = \
                len(data_loader) * self.batchsize // self._checkpoint_batchsize
            checkpoint_inner_iter_left = checkpoint_data_loader_len - self._checkpoint_inner_iter
            inner_iter_left = \
                checkpoint_inner_iter_left * self._checkpoint_batchsize // self.batchsize
            if inner_iter_left < 0:
                self.logger.warn("Data loader length {} < inner iter {} of checkpoint".format(
                    checkpoint_data_loader_len, self._checkpoint_inner_iter))
                if inner_iter_left == -1:
                    inner_iter_left = 0
                    self.logger.warn("We assume the last epoch has finished."
                                     "Start from new epoch.")
                else:
                    inner_iter_left = len(data_loader)
            self._checkpoint_batchsize = None

        do_seach = False
        search_data_iter = None
        if self.search_optimizer is not None:
            assert hasattr(self, 'tune_epoch_start') and hasattr(self, 'tune_epoch_end')
            do_seach = self.tune_epoch_end >= self.epoch + 1 >= self.tune_epoch_start
            if do_seach:
                search_data_iter = iter(self.search_data_loader)
                self.model.module.reset_do_search(True)
            else:
                self.model.module.reset_do_search(False)

        self._max_inner_iter = len(data_loader)
        set_total_inner_iter(self._max_inner_iter)
        self.call_hook('before_train_epoch')
        # alternate training params and arch-params
        bar = tqdm(islice(enumerate(data_loader), 0, inner_iter_left),
                   total=inner_iter_left, ncols=70)
        for i, data_batch in bar:
            self._inner_iter = i + len(data_loader) - inner_iter_left
            set_inner_iter(self._inner_iter)

            bar_desc = getattr(self, 'task_name', 'none')
            if do_seach:
                bar_desc = bar_desc + " Searching"
                try:
                    search_data_batch = next(search_data_iter)
                except StopIteration:
                    search_data_iter = iter(self.search_data_loader)
                    search_data_batch = next(search_data_iter)
                search_outputs = self.batch_processor(
                    self.model, search_data_batch, train_mode=True, **kwargs)
                self.search_outputs = search_outputs
                self.call_hook('after_val_iter')

            bar.set_description(bar_desc)
            self.call_hook('before_train_iter')
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=True, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
                self.log_buffer.update(get_summary())
            self.outputs = outputs
            self.call_hook('after_train_iter')
            update_global_step(self.batchsize)
            self._iter += 1

        self.call_hook('after_train_epoch')
        if do_seach:
            # it may cause undefined behavior if mmdetection adds hooks with 'after_val_epoch'
            self.call_hook('after_val_epoch')
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                outputs = self.batch_processor(
                    self.model, data_batch, train_mode=False, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch'] - 1
        self._iter = checkpoint['meta']['iter']  # -1 is not needed.
        self._checkpoint_inner_iter = checkpoint['meta'].get('inner_iter', None)
        self._checkpoint_batchsize = checkpoint['meta'].get('batchsize', None)
        self._checkpoint_global_step = checkpoint['meta'].get('global_step', None)

        if self._checkpoint_global_step:
            reset_global_step(self._checkpoint_global_step, logger=self.logger)
        else:
            self.logger.warn("Can NOT find key 'global_step' in checkpoint['meta'], you may "
                             "meet misalign in tensorboard if batchsize changed.")

        if 'optimizer' in checkpoint and resume_optimizer:
            checkpoint_initial_lr = checkpoint['meta'].get('initial_lr', None)
            if checkpoint_initial_lr and self._checkpoint_batchsize:
                resume_initial_lr = \
                    checkpoint_initial_lr / self._checkpoint_batchsize * self.batchsize
                for i in range(len(checkpoint['optimizer']['param_groups'])):
                    checkpoint['optimizer']['param_groups'][i]['initial_lr'] = resume_initial_lr
            else:
                self.logger.warn("Can NOT automatically reset lr, you may need to set lr manually.")

            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch + 1, self.iter)

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow) or self.search_optimizer is not None

        set_total_epoch(max_epochs)
        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
        self.call_hook('before_run')

        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            'runner has no method named "{}" to run an epoch'.
                            format(mode))
                    epoch_runner = getattr(self, mode)
                elif callable(mode):  # custom train()
                    epoch_runner = mode
                else:
                    raise TypeError('mode in workflow must be a str or '
                                    'callable function, not {}'.format(
                                        type(mode)))
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= max_epochs:
                        return
                    # replace logger.info
                    logger_info = self.logger.info
                    self.logger.info = self.logger.tqdm_write
                    set_epoch(self.epoch + 1)

                    if self.search_optimizer is not None and mode == 'train':
                        assert len(workflow) == 1
                        epoch_runner(data_loaders, **kwargs)
                    else:
                        epoch_runner(data_loaders[i], **kwargs)

                    # recover logge.infor
                    self.logger.info = logger_info

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def register_lr_hooks(self, lr_config):
        if isinstance(lr_config, LrUpdaterHook):
            self.register_hook(lr_config)
        elif isinstance(lr_config, dict):
            assert 'policy' in lr_config
            # from .hooks import lr_updater
            hook_name = lr_config['policy'].title() + 'LrUpdaterHook'
            if not hasattr(lr_updater, hook_name):
                raise ValueError('"{}" does not exist'.format(hook_name))
            hook_cls = getattr(lr_updater, hook_name)
            self.register_hook(hook_cls(**lr_config))
        else:
            raise TypeError('"lr_config" must be either a LrUpdaterHook object'
                            ' or dict, not {}'.format(type(lr_config)))

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - OptimizerStepperHook
        - IterTimerHook
        """
        if optimizer_config is None:
            optimizer_config = {}
        self.register_lr_hooks(lr_config)
        self.register_hook(self.build_hook(optimizer_config, OptimizerHook))
        self.register_hook(IterTimerHook())
