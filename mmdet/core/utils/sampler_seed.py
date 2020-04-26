from mmcv.runner.hooks import Hook


class DistSamplerSeedHook(Hook):

    def before_epoch(self, runner):
        if isinstance(runner.data_loader, list):
            for d in runner.data_loader:
                d.sampler.set_epoch(runner.epoch)
        else:
            runner.data_loader.sampler.set_epoch(runner.epoch)
