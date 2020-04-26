import numpy as np
import random
import torch
import cv2
import math
from PIL import Image
import torch.nn.functional as F
import onnx
import onnxruntime as rt

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result, raw_inference
from mmdet.core.utils.model_utils import describe_vars
from tools.model_zoo import model_zoo as zoo

version = 'cm-v0.9'
cfg = mmcv.Config.fromfile(zoo[version]['config'])
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
model_file = zoo[version]['model_file']
model.load_state_dict(torch.load(model_file)['state_dict'], strict=True)

for m in model.modules():
    m.eval()
model.eval()
model.cuda()

# export to onnx
output_names = zoo[version]['output']
onnx_file = model_file.replace('.pth', '.onnx')
assert (model_file != onnx_file), onnx_file
dummpy_input = torch.randn(1, 3, 800, 1280, device='cuda')
torch.onnx.export(
    model,
    dummpy_input,
    onnx_file,
    verbose=True,
    input_names=['data'],
    output_names=output_names)
print('export to {} done!'.format(onnx_file))

print("check onnx model")
onnx_model = onnx.load(onnx_file)
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))

# inference
sess = rt.InferenceSession(onnx_file)

inputs = sess.get_inputs()
for i in range(len(inputs)):
    print('input', i, inputs[i].name, inputs[i].shape, inputs[i].type)

outputs = sess.get_outputs()

for i in range(len(outputs)):
    print('output:', i, outputs[i].name, outputs[i].shape, outputs[i].type)
    # output_names.append(outputs[i].name)

x = np.random.random(inputs[0].shape)
x = x.astype(np.float32)

results = sess.run(output_names, {inputs[0].name: x})

print(len(results))
print('export to {}'.format(onnx_file))
