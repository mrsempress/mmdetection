import yaml
import json
from termcolor import colored

with open('./watch_dog/dog_cfg.yml') as f:
    dog_cfg = yaml.safe_load(f)

for task_name in dog_cfg['TASKS']:
    locals()[task_name] = dog_cfg['TASKS'][task_name]

for device_name in dog_cfg['DEVICES']:
    locals()[device_name] = dog_cfg['DEVICES'][device_name]

name_to_cfg = {
    'liuzili-System-Product-Name': 'MYGPU',
    'gpu3.fabu.ai': 'GPU3',
    'gpu4.fabu.ai': 'GPU4',
    'gpu7.fabu.ai': 'GPU7',
    'gpu9.fabu.ai': 'GPU9',
    'gpu10.fabu.ai': 'GPU10',
    'gpu11.fabu.ai': 'GPU11',
    'gpu12.fabu.ai': 'GPU12',
    'gpu13.fabu.ai': 'GPU13',
    'gpu14.fabu.ai': 'GPU14',
}


def cfg_process(cfg):
    cfg = eval(cfg)
    cfg['tasks'] = [eval(cfg['tasks'][i]) for i in range(len(cfg['tasks']))]
    cfg['watch_gpus'] = list(cfg['watch_gpus'])
    return cfg


def get_config(hostname):
    cfg = cfg_process(name_to_cfg[hostname])
    print(colored("Get config of {}:".format(hostname), 'cyan'))
    print(colored(json.dumps(cfg, indent=4), 'green'))
    return cfg
