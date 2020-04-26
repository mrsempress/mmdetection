import os
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
from tqdm import tqdm
import pickle as pkl
import mmcv
from mmdet.ops import nms

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
    bindings = []
    bind_names = []
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
            bind_names.append(binding)
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, bind_names, stream


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


def postprocess(wh_feats, reg_offsets, heatmaps, heatmap_indexs, rf):
    score_thresh = 0.01
    pool_scale = 4 * 1.5
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
    scores = scores[..., np.newaxis]
    bboxes = np.concatenate((bboxes, scores), axis=2)
    return bboxes, labels


def get_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def get_files():
    data_dir = '/private/ningqingqun/obstacle_detector/data/obstacle2d/'
    split_file = os.path.join(data_dir, 'ImageSets', 'val1010.txt')
    with open(split_file) as f:
        name_list = f.read().splitlines()

    im_list = [
        os.path.join(data_dir, 'JPGImages', l + '.jpg') for l in name_list
    ]
    print('get {} images'.format(len(im_list)))

    return im_list


def get_files2():
    data_dir = '/private/ningqingqun/bags/truck1_2019_08_01_15_21_03_19.msg/front_right/201908011521/'
    im_list = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith('.jpg')
    ]

    return im_list


def eval_split():
    file_list = get_files()
    engine_file_path = '/private/ningqingqun/torch/centernet/r34_epoch_24_iter_126179-6.1.trt'
    results = []
    num_fg = 7
    topk = 50
    class_names = [
        'car', 'bus', 'truck', 'person', 'bicycle', 'tricycle', 'block'
    ]
    out_dir = '/private/ningqingqun/datasets/centernet_results/trt'
    input_h, input_w = (800, 1280)
    output_h = int(input_h / 4)
    output_w = int(input_w / 4)
    out_shapes = {
        'heatmap': (1, num_fg, output_h, output_w),
        'heatmap_indexs': (1, topk),
        'reg_offset': (1, 2, output_h, output_w),
        'wh_feats': (1, 2, output_h, output_w),
        'raw_features': (1, output_h, output_w, 64),
    }
    with get_engine(engine_file_path
                    ) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, out_names, stream = allocate_buffers(engine)
        print(bindings)
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        for imfile in tqdm(file_list, total=len(file_list), ncols=80):
            im = cv2.imread(imfile)

            resized_image = cv2.resize(im, (input_w, input_h))
            input_image = preprocess(resized_image)
            inputs[0].host = input_image
            trt_outputs = do_inference(
                context,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream)
            trt_outputs = [
                output.reshape(out_shapes[bname])
                for output, bname in zip(trt_outputs, out_names)
            ]

            bboxes, labels = postprocess(*trt_outputs)
            bboxes = bboxes[0]
            labels = labels[0]
            bboxes, nms_idx = nms(bboxes, 0.5)
            labels = labels[nms_idx]

            # print(imfile)
            # output_image_path = 'eval_trt.png'
            #
            # output_image_path = os.path.join(out_dir, os.path.basename(imfile))
            # mmcv.imshow_det_bboxes(
            #     im.copy(),
            #     bboxes,
            #     labels,
            #     class_names=class_names,
            #     score_thr=0.3,
            #     show=output_image_path is None,
            #     out_file=output_image_path)
            # break

            detections = [[] for _ in range(num_fg)]
            for i in range(len(labels)):
                c = labels[i]
                detections[c].append(bboxes[i])
            results.append(detections)

    result_file = engine_file_path.replace('.trt', '_correct.pkl')
    pkl.dump(results, open(result_file, 'wb'), protocol=2)
    print('result saved to {}'.format(result_file))


def eval_speed():
    imfile = '/private/ningqingqun/bags/crane/howo1_2019_11_05_09_06_22_6.msg/front_right/201911050907/201911050907_00000417_1572916023857.jpg'
    engine_file_path = '/private/ningqingqun/torch/eval_time/1g/vision_detector_fabu_v4.1.1-5.1.5.0-6.1.trt'
    engine_file_path = '/private/ningqingqun/torch/eval_time/4g/vision_detector_fabu_v4.1.1-5.1.5.0-6.1.trt'

    input_h, input_w = (800, 1280)
    repeat_time = 1000
    perf_times = []
    proc_times = []
    past_times = []
    with get_engine(engine_file_path
                    ) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, out_names, stream = allocate_buffers(engine)

        im = cv2.imread(imfile)
        resized_image = cv2.resize(im, (input_w, input_h))
        input_image = preprocess(resized_image)
        inputs[0].host = input_image

        perf_start = time.perf_counter()
        proc_start = time.process_time()
        past_start = time.time()
        for i in range(repeat_time):
            trt_outputs = do_inference(
                context,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream)
            if i < 5:
                print(time.perf_counter() - perf_start)
        print('perf coutner time: {}ms'.format(
            (time.perf_counter() - perf_start) * 1000 / repeat_time))
        print('perf coutner time: {}ms'.format(
            (time.process_time() - proc_start) * 1000 / repeat_time))
        print('perf coutner time: {:.2f}ms'.format(
            (time.time() - past_start) * 1000 / repeat_time))


if __name__ == '__main__':
    eval_speed()
