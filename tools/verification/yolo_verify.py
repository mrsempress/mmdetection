import numpy as np
import torch
import cv2
import math
from PIL import Image

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result, raw_inference
from mmdet.core.utils.model_utils import describe_vars

cfg = mmcv.Config.fromfile('configs/yolo/yolov3_d53.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
describe_vars(model)

"""Transfer the params

params = {}
for name, p in model.state_dict().items():
    if name.startswith('bbox_head.yolo_blocks.'):
        print(name[22:])
        name = 'neck.darknet_fpn_block.' + name[len('bbox_head.yolo_blocks.'):]
    elif name.startswith('bbox_head.transitions.'):
        name = 'neck.transitions.' + name[len('bbox_head.transitions.'):]
    params[name] = p.cpu().numpy()
    print(params[name].dtype)
np.savez('yolov3.npz', params)

params = np.load('./yolov3.npz')
param_files = {}
for file in params.files:
    param_files[file] = params[file]

new_params = {}
valid_i = 0
for param_tensor in model.state_dict():
    new_params[param_tensor] = torch.tensor(param_files[param_tensor], dtype=torch.float32)
    valid_i += 1

assert valid_i == len(param_files)
model.load_state_dict(new_params)
torch.save(model.state_dict(), 'yolov3.pth')
"""

model.load_state_dict(torch.load('yolov3.pth'), strict=True)
for m in model.modules():
    m.eval()
model.cuda()

# x = np.load('../gluon-cv/yolo_input.npy')
x = cv2.imread('./data/coco/val2017/000000135872.jpg')
resize = int(math.ceil(512 / 0.875))
ratio = x.shape[0] / x.shape[1]
new_size = (int(resize * ratio), resize) if ratio > 1 else (resize, int(resize / ratio))
x = cv2.resize(x, (new_size[1], new_size[0]))
x = np.array(x)[:, :, ::-1]

nimg = Image.fromarray(np.array(x))
from torchvision import transforms

nimg = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])(nimg)

nimg = nimg.view(1, *nimg.size())
result = model([nimg.cuda()], return_loss=False)
# x = np.squeeze(x, 0).transpose(1, 2, 0)
show_result(x, result, out_file='./tmp1.jpg')

# img = np.zeros((3, 256, 256), dtype=np.float32)
#
# start = 0
# stride = 4
#
# img[:, start:start+stride, start:start+stride] = 1.
# result = raw_inference(model, img, device='cuda:1')

'''Transfer the mxnet yolov3 params.

params = np.load('../gluon-cv/yolov3.npz')
param_files = []
for file in params.files:
    if 'anchor' in file or 'offset' in file:
        continue
    param_files.append(params[file])

new_params = {}
valid_i = 0
for param_tensor in model.state_dict():
    if 'track' in param_tensor:
        continue
    new_params[param_tensor] = torch.tensor(param_files[valid_i], dtype=torch.float32)
    valid_i += 1

assert valid_i == len(param_files)
model.load_state_dict(new_params)
torch.save(model.state_dict(), 'yolov3.pkl')
'''
