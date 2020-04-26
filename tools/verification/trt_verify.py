from __future__ import print_function

import os
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import mmcv
from tqdm import tqdm
import pickle as pkl

from vis_util import show_corners
from tools.model_zoo import model_zoo as zoo

TRT_LOGGER = trt.Logger()


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    output_names = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(
            engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        print('binding:{}, size:{}, dtype:{}'.format(binding, size, dtype))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            output_names.append(binding)
    return inputs, outputs, output_names, bindings, stream


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(
        batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(builder):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with builder.create_network() as network, trt.OnnxParser(
                network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 27  # 1GB
            builder.max_batch_size = 1
            print('max workspace size: {:.2f} MB'.format(
                builder.max_workspace_size / 1024 / 1024))
            tic = time.time()

            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please generate it.'.format(
                    onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.
                  format(onnx_file_path))
            engine = builder.build_cuda_engine(network)

            if engine is None:
                raise Exception('build engine failed')
            else:
                print('Completed! time cost: {:.1f}s'.format(time.time() -
                                                             tic))
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    with trt.Builder(TRT_LOGGER) as builder:
        if builder.platform_has_fast_fp16:
            print('enable fp16 mode!')
            builder.fp16_mode = True
            builder.strict_type_constraints = True
            engine_file_path = engine_file_path.replace('.trt', '_fp16.trt')

        if os.path.exists(engine_file_path):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path,
                      "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return build_engine(builder)


def preprocess(image, use_rgb=True):
    image = image.astype(np.float32)
    mean_rgb = np.array([123.675, 116.28, 103.53])
    std_rgb = np.array([58.395, 57.12, 57.375])

    if use_rgb:
        image = image[..., [2, 1, 0]]
        image -= mean_rgb
        image /= std_rgb
    else:
        mean_bgr = mean_rgb[[2, 1, 0]]
        std_bgr = std_rgb[[2, 1, 0]]
        image -= mean_bgr
        image /= std_bgr

    image = np.transpose(image, [2, 0, 1])
    return np.ascontiguousarray(image)


def postprocess_2d(wh_feats, reg_offsets, heatmaps, heatmap_indexs, rf):
    score_thresh = 0.01
    h, w = heatmaps.shape[-2:]
    batch, topk = heatmap_indexs.shape

    heatmaps = heatmaps.reshape((batch, -1))
    scores = np.take(heatmaps, heatmap_indexs)

    labels = (heatmap_indexs // (h * w)).astype(int)

    spatial_idx = heatmap_indexs % (h * w)
    offsetx = np.take(reg_offsets[:, 0, ...].reshape((batch, -1)), spatial_idx)
    offsety = np.take(reg_offsets[:, 1, ...].reshape((batch, -1)), spatial_idx)

    pred_w = np.take(wh_feats[:, 0, ...].reshape((batch, -1)), spatial_idx)
    pred_h = np.take(wh_feats[:, 1, ...].reshape((batch, -1)), spatial_idx)

    cx = spatial_idx % w + offsetx
    cy = spatial_idx // w + offsety

    x1 = cx - pred_w / 2
    y1 = cy - pred_h / 2
    x2 = cx + pred_w / 2
    y2 = cy + pred_h / 2
    bboxes = np.stack([x1, y1, x2, y2], axis=2) * pool_scale

    return bboxes, labels, scores


def show_results_2d(img, outputs, output_image_path, class_names):
    # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
    bboxes, labels, scores = postprocess_2d(*outputs)
    scores = scores[..., np.newaxis]
    bboxes = np.concatenate((bboxes, scores), axis=2)
    mmcv.imshow_det_bboxes(
        img,
        bboxes[0],
        labels[0],
        class_names=class_names,
        score_thr=0.35,
        show=output_image_path is None,
        out_file=output_image_path)
    # Draw the bounding boxes onto the original input image and save it as a PNG file
    # obj_detected_img = draw_bboxes(resized_image, boxes[0], scores[0], classes[0])
    #
    # cv2.imwrite(output_image_path, obj_detected_img)
    # print('Saved image with bounding boxes of detected objects to {}.'.format(
    # output_image_path))


def postprocess_3d(heatmaps, height_feats, reg_xoffsets, reg_yoffsets, poses,
                   heatmap_indexs):
    batch, _, h, w = heatmaps.shape

    results = []
    for i in range(batch):
        idxs = heatmap_indexs[i]
        scores = heatmaps[i].reshape(-1)[idxs]

        labels = (idxs // (h * w)).astype(int)
        idxs = idxs % (h * w)
        x_idxs = idxs % w
        y_idxs = idxs // w
        offsetx = reg_xoffsets[i, :, y_idxs, x_idxs]
        offsety = reg_yoffsets[i, :, y_idxs, x_idxs]
        height = height_feats[i, :, y_idxs, x_idxs]
        poses = poses[i, y_idxs, x_idxs]

        cx = x_idxs[:, np.newaxis] + offsetx
        cy = y_idxs[:, np.newaxis] + offsety
        cy1 = cy - height / 2
        cy2 = cy + height / 2
        corners = np.stack([np.hstack([cx, cx]), np.hstack([cy1, cy2])])
        corners = np.transpose(corners, [1, 2, 0]) * pool_scale
        pose_scores = np.zeros_like(poses)
        results.append([corners, labels, scores, poses, pose_scores])

    return results


def show_results_3d(img, outputs, output_image_path, class_names):
    height_feats, reg_xoffsets, reg_yoffsets, poses, heatmaps, heatmap_indexs, _ = outputs
    results = postprocess_3d(heatmaps, height_feats, reg_xoffsets,
                             reg_yoffsets, poses, heatmap_indexs)
    score_thresh = 0.35
    with open('/private/ningqingqun/undistort.pkl', 'wb') as f:
        pkl.dump(results[0][0][0], f)
    show_corners(
        img,
        results[0],
        class_names,
        score_thr=score_thresh,
        out_file=output_image_path,
        pad=0)


def get_images2():
    im_list = [
        '/private/ningqingqun/datasets/undist_img.png'
        # '/private/ningqingqun/datasets/outsource/201910110946_00000187_1570758388348.jpg'
    ]

    return im_list


def get_images():
    # input_image_path = '/private/ningqingqun/bags/truck1_2019_07_24_14_54_43_26.msg/front_right/201907241454/201907241454_00000000_1563951283641.jpg'
    # input_image_path = '/private/ningqingqun/bags/truck2_2019_07_26_17_02_47_1.msg/front_right/201907261702/201907261702_00000002_1564131768223.jpg'
    # input_image_path = '/private/ningqingqun/bags/truck1_2019_09_06_13_48_14_19.msg/front_right/201909061348/201909061348_00000005_1567748895027.jpg'
    # data_dir = '/private/ningqingqun/bags/crane/howo1_2019_12_04_09_14_54_0.msg/front_left/201912040915'
    # data_dir = '/private/ningqingqun/bags/howo1_2019_12_11_08_59_10_6.msg/head_right/201912110859'
    # data_dir = '/private/ningqingqun/bags/jinlv4_2019_10_18_09_18_50_6.msg/head_right/201910180919'
    data_dir = '/private/ningqingqun/bags/howo1_2019_12_24_17_49_48_2.msg/front_left/201912241750'
    # data_dir = '//private/ningqingqun/datasets/outsource/mine/truck2/front_right/20191220'
    im_list = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith('.jpg')
    ]
    return im_list


# Output shapes expected by the post-processor
version = 'v5.5.2'
if 'cm' in version:
    num_fg = 12
else:
    num_fg = 7
topk = 50
input_h, input_w = (800, 1280)
out_channels = 64
pool_scale = 4
output_h = int(input_h / pool_scale)
output_w = int(input_w / pool_scale)

onnx_files = {
    'v4_fp16':
    '/private/ningqingqun/torch/centernet/r34_fp16_epoch_16_iter_60000.onnx',
    'v5.1.16':
    '/private/ningqingqun/mmdet/outputs/v5.1.16/centernet_r18_ignore_1017_1915_gpu12/epoch_35_iter_3675.onnx',
    'v5.tmp':
    'work_dirs/debug/centernet_r18_ignore_1105_1118_desktop/epoch_1_iter_500.onnx',
    'cm-v0.1':
    'work_dirs/debug/centernet_r18_no_1119_1954_desktop/epoch_35_iter_4305.onnx',
    'cm-v0.2':
    'work_dirs/debug/centernet_r18_no_1120_1157_desktop/epoch_40_iter_4920.onnx',
    'cm-v0.6':
    '/private/ningqingqun/mmdet/outputs/no31_36/centernet_r18_adam_no_crop_1129_1920_gpu9/epoch_10_iter_2000.onnx',
    'cm-v0.8':
    '/work/work_dirs/v5.3.3/centernet_r18_finetune_large_1207_1707_desktop/epoch_20_iter_1160.onnx'
}
name2shape = {
    'heatmap': (1, num_fg, output_h, output_w),
    'height_feats': (1, 3, output_h, output_w),
    'reg_xoffset': (1, 3, output_h, output_w),
    'reg_yoffset': (1, 3, output_h, output_w),
    'pose': (1, output_h, output_w),
    'raw_features': (1, output_h, output_w, out_channels),
    'heatmap_indexs': (1, topk),
    'wh_feats': (1, 2, output_h, output_w),
    'reg_offset': (1, 2, output_h, output_w),
}


def main():
    """Create a TensorRT engine for ONNX-based centernet and run inference."""

    try:
        cuda.init()
        major, minor = cuda.Device(0).compute_capability()
    except:
        raise Exception("failed to get gpu compute capability")

    onnx_file_path = zoo[version]['model_file'].replace('.pth', '.onnx')
    new_ext = '-{}.{}.trt'.format(major, minor)
    engine_file_path = onnx_file_path.replace('.onnx', new_ext)

    # engine_file_path ='/private/ningqingqun/torch/centernet/vision_detector_fabu_v4.0.0-5.1.5.0-6.1.trt'
    # Download a dog image and save it to the following file path:
    image_list = get_images()
    out_dir = '/private/ningqingqun/results/trt_results/' + version + '_20191220_mining'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Do inference with TensorRT
    trt_outputs = []
    with get_engine(onnx_file_path, engine_file_path
                    ) as engine, engine.create_execution_context() as context:
        inputs, outputs, output_names, bindings, stream = allocate_buffers(
            engine)
        # Do inference
        # print('Running inference on image {}...'.format(input_image_path))
        # Set host input to the image.
        # The common.do_inference function will copy the input to the GPU
        # before executing.
        for input_image_path in tqdm(image_list):
            # input_h, input_w = (input_h // 32 * 32, input_w // 32 * 32)
            im = cv2.imread(input_image_path)
            resized_image = cv2.resize(im, (input_w, input_h))
            input_image = preprocess(resized_image)
            inputs[0].host = input_image
            # tic = time.time()
            trt_outputs = do_inference(
                context,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream)
            # print('inference time cost: {:.1f}ms'.format(
            # (time.time() - tic) * 1000))

            # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
            trt_outputs = [
                output.reshape(name2shape[name])
                for output, name in zip(trt_outputs, output_names)
            ]

            class_names = [
                'car', 'bus', 'truck', 'person', 'bicycle', 'tricycle', 'block'
            ]
            out_file = os.path.join(out_dir,
                                    os.path.basename(input_image_path))
            if 'v5' in version:
                show_results_3d(resized_image.copy(), trt_outputs, out_file,
                                class_names)
            elif 'cm' in version:
                class_names = [
                    'right20',
                    'right40',
                    'right45',
                    'left20',
                    'left40',
                    'left45',
                    'NO31',
                    'NO32',
                    'NO33',
                    'NO34',
                    'NO35',
                    'NO36',
                ]
                show_results_2d(resized_image.copy(), trt_outputs, out_file,
                                class_names)
            else:
                show_results_2d(resized_image.copy(), trt_outputs, out_file,
                                class_names)


if __name__ == '__main__':
    main()
