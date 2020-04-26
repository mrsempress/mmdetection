from mmcv.runner.hooks import Hook
from mmcv.runner import master_only

from .summary import add_histogram_summary


def _make_hook(model, model_types):
    out_feats = {}
    hooks = []

    def _mhook(name):

        def _hook(m, input, output):
            out_feats[name] = output.detach().cpu()

        return _hook

    for name, m in model.named_modules():
        if type(m).__name__ in model_types:
            hook = m.register_forward_hook(_mhook(name))
            hooks.append(hook)

    return out_feats, hooks


class HistogramHook(Hook):

    def __init__(self, name, model_types, save_every_n_steps, sub_modules=None):
        self.name = name
        self.model_types = model_types
        self.save_every_n_steps = save_every_n_steps
        self.sub_modules = sub_modules

        self.out_feats = {}
        self.hooks = []

    @master_only
    def before_run(self, runner):
        model = runner.model
        if self.sub_modules is not None:
            if hasattr(model, 'module'):
                model = model.module
            for sub_module in self.sub_modules:
                assert hasattr(model, sub_module), \
                    "Can't find {} ({}) in model.".format(sub_module, self.sub_modules)
                model = getattr(model, sub_module)

    @master_only
    def before_train_iter(self, runner):
        model = runner.model
        if self.sub_modules is not None:
            if hasattr(model, 'module'):
                model = model.module
            for sub_module in self.sub_modules:
                model = getattr(model, sub_module)

        if self.every_n_iters(runner, self.save_every_n_steps):
            self.out_feats, self.hooks = _make_hook(model, self.model_types)

    @master_only
    def after_train_iter(self, runner):
        # note that when after_train_iter is called, the iter num has not plus 1 yet.
        if self.every_n_iters(runner, self.save_every_n_steps):
            for k, v in self.out_feats.items():
                add_histogram_summary("{}/{}".format(self.name, k), v)

            for hook in self.hooks:
                hook.remove()
