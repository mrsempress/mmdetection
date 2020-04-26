import numpy as np
import random
import torch
import cv2
import math
from PIL import Image
import torch.nn.functional as F

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result, raw_inference
from mmdet.core.utils.model_utils import describe_vars

cfg = mmcv.Config.fromfile('configs/fcos/fcos_r50_1x.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
describe_vars(model)

""" Transfer the fcos params.

params = np.load('../FCOS/fcos_r50.npz')
param_files = []
for file in params.files:
    param_files.append(params[file])

new_params = {}
valid_i = 0
for param_tensor in model.state_dict():
    if 'track' in param_tensor:
        continue
    suffix = param_tensor.split('.')[-1]
    if param_tensor.startswith('neck.lateral_convs.1.conv'):
        new_params[param_tensor] = torch.tensor(
            params['backbone.fpn.fpn_inner3.' + suffix], dtype=torch.float32)
    elif param_tensor.startswith('neck.lateral_convs.2.conv'):
        new_params[param_tensor] = torch.tensor(
            params['backbone.fpn.fpn_inner4.' + suffix], dtype=torch.float32)
    elif param_tensor.startswith('neck.fpn_convs.0.conv'):
        new_params[param_tensor] = torch.tensor(
            params['backbone.fpn.fpn_layer2.' + suffix], dtype=torch.float32)
    elif param_tensor.startswith('neck.fpn_convs.1.conv'):
        new_params[param_tensor] = torch.tensor(
            params['backbone.fpn.fpn_layer3.' + suffix], dtype=torch.float32)
    else:
        new_params[param_tensor] = torch.tensor(param_files[valid_i], dtype=torch.float32)
    valid_i += 1

assert valid_i == len(param_files)
model.load_state_dict(new_params)
torch.save(model.state_dict(), 'fcos_r50.pkl')
"""

model.load_state_dict(torch.load('./pretrain/fcos_fpn_r50.pth'), strict=True)
for m in model.modules():
    m.eval()
model.cuda()

x = np.load('./img.npy')
x1 = cv2.imread('./data/coco/val2017/000000000139.jpg')

#resize = int(math.ceil(800 / 0.875))
#ratio = x.shape[0] / x.shape[1]
#new_size = (int(resize * ratio), resize) if ratio > 1 else (resize, int(resize / ratio))
#new_size = (new_size[0] // 32 * 32, new_size[1] // 32 * 32)
x1 = cv2.resize(x1, (x.shape[3], x.shape[2]))
# x = np.array(x)[:, :, ::-1]
# nimg = Image.fromarray(np.array(x))
from torchvision.transforms import functional as F1
x2 = x1.astype(np.float32)
mean_bgr = [103.53, 116.28, 123.675]
x2 -= mean_bgr
x2 = np.transpose(x2, [2, 0, 1])

#from torchvision import transforms

#nimg = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#])(nimg)
nimg = torch.tensor(x2)
nimg = nimg.view(1, *nimg.size())
result = model([torch.tensor(x).cuda()], return_loss=False)
# x = np.squeeze(x, 0).transpose(1, 2, 0)
show_result(x1, result, out_file='./tmp1.jpg')

# img = np.zeros((3, 256, 256), dtype=np.float32)
#
# start = 0
# stride = 4
#
# img[:, start:start+stride, start:start+stride] = 1.
# result = raw_inference(model, img, device='cuda:1')
