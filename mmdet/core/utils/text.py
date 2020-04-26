import datetime
from collections import defaultdict
from termcolor import colored

from mmcv.runner.hooks.logger.base import LoggerHook


class TextLoggerHook(LoggerHook):

    def __init__(self, interval, ignore_last=True, reset_flag=False):
        super(TextLoggerHook, self).__init__(interval, ignore_last, reset_flag)
        self.time_sec_tot = 0

    def before_run(self, runner):
        super(TextLoggerHook, self).before_run(runner)
        self.start_iter = runner.iter

    def log(self, runner):
        log_str = colored('[{}] '.format(
            datetime.datetime.now().strftime('%m%d %H:%M:%S')), 'green')

        if runner.mode == 'train':
            lr_str = colored(', lr: {}'.format(', '.join(
                ['{:.5f}'.format(lr) for lr in runner.current_lr()[:2]])), 'cyan')
            log_str += colored('[Epoch {}][{}/{}]\t\t'.format(
                runner.epoch + 1, runner.inner_iter + 1, len(runner.data_loader)), 'green')
        else:
            lr_str = ''
            log_str += colored('[Epoch({})] [{}][{}]\t\t'.format(runner.mode, runner.epoch,
                                                                 runner.inner_iter + 1), 'green')
        if 'time' in runner.log_buffer.output:
            self.time_sec_tot += (runner.log_buffer.output['time'] * self.interval)
            time_sec_avg = self.time_sec_tot / (runner.iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            log_str += colored('eta: {}, '.format(eta_str), 'green')
            log_str += colored('time: {:.3f}, data_time: {:.3f}'.
                               format(runner.log_buffer.output['time'] / runner.imgs_per_gpu,
                                      runner.log_buffer.output['data_time']), 'green')
        log_str += lr_str

        log_dict = defaultdict(dict)
        lprefix_dict = defaultdict(list)
        for name, val in runner.log_buffer.output.items():
            if name in ['time', 'data_time']:
                continue
            if '/' not in name:
                log_str += colored(', {}: {:.3f}'.format(name, val), 'cyan')
            else:
                prefix = name.split('/')[0]
                suffix = '/'.join(name.split('/')[1:])
                # divide the suffix, e.g. num_fgs => num, fgs.
                suffix_pre = '_'.join(suffix.split('_')[:-1])
                suffix_su = suffix.split('_')[-1]
                lprefix = '{}/{}'.format(prefix, suffix_pre)
                lprefix_dict[lprefix].append(suffix_su)
                log_dict[prefix][suffix] = val

        log_str += colored(', gpus: {}, imgs_per_gpu: {}'.format(
            runner.world_size, runner.imgs_per_gpu), 'yellow')

        is_choosed = []
        for prefix, log_d in log_dict.items():
            log_str += colored('\n\t\t\t\t\t\t{}: \n\t\t\t\t\t\t\t'.format(prefix), 'blue')
            log_items = []  # recode a line.
            for suffix, val in log_d.items():
                if '{}/{}'.format(prefix, suffix) in is_choosed:
                    continue
                suffix_pre = '_'.join(suffix.split('_')[:-1])
                lprefix = '{}/{}'.format(prefix, suffix_pre)
                if suffix_pre == '' or len(lprefix_dict[lprefix]) == 1:
                    line_head = '{}:'.format(suffix).ljust(40)
                    line_content = '{:.3f}'.format(val)
                    is_choosed.append('{}/{}'.format(prefix, suffix))
                else:
                    line_head = '{}({}):'.format(
                        suffix_pre, '/'.join(lprefix_dict[lprefix])).ljust(40)
                    list_item = []
                    for suffix_su in lprefix_dict[lprefix]:
                        list_item.append(
                            '{:.3f}'.format(log_d['{}_{}'.format(suffix_pre, suffix_su)]))
                        is_choosed.append('{}/{}_{}'.format(prefix, suffix_pre, suffix_su))
                    line_content = '[{}]'.format(', '.join(list_item))
                log_items.append(line_head + line_content)
            log_str += '\n\t\t\t\t\t\t\t'.join(log_items)
        runner.logger.info(log_str)
