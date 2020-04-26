from tabulate import tabulate
from termcolor import colored
from graphviz import Digraph
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

from mmdet.core.utils import logger


def describe_vars(model):
    headers = ['name', 'shape', '#elements', '(M)', '(MB)', 'trainable', 'dtype']

    data = []
    trainable_count = 0
    total = 0
    total_size = 0
    trainable_total = 0

    for name, param in model.named_parameters():
        dtype = str(param.data.dtype)
        param_mb = 'NAN'
        param_bype = float(''.join([s for s in dtype if s.isdigit()])) / 8

        total += param.data.nelement()
        if param_bype:
            param_mb = '{:.02f}'.format(param.data.nelement() / 1024.0 ** 2 * param_bype)
            total_size += param.data.nelement() * param_bype
        data.append([name.replace('.', '/'), list(param.size()),
                     '{0:,}'.format(param.data.nelement()),
                     '{:.02f}'.format(param.data.nelement() / 1024.0 ** 2), param_mb,
                     param.requires_grad, dtype])
        if param.requires_grad:
            trainable_count += 1
            trainable_total += param.data.nelement()

    table = tabulate(data, headers=headers)

    summary_msg = colored(
        "\nNumber of All variables: {}".format(len(data)) +
        "\nAll parameters (elements): {:.02f}M".format(total / 1024.0 ** 2) +
        "\nAll parameters (size): {:.02f}MB".format(total_size / 1024.0 ** 2) +
        "\nNumber of Trainable variables: {}".format(trainable_count) +
        "\nAll trainable parameters (elements): {:.02f}M\n".format(
            trainable_total / 1024.0 ** 2), 'cyan')
    logger.info(colored("List of All Variables: \n", 'cyan') + table + summary_msg)


def describe_features(model, input_size=(3, 800, 800)):
    headers = ['type', 'input', 'output', '#elements(M)', '(MB)', 'dtype', '#params(M)']

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = "%s-%i" % (class_name, module_idx + 1)

            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = -1
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = -1

            dtype = str(output.dtype)
            elements_byte = float(''.join([s for s in dtype if s.isdigit()])) / 8
            summary[m_key]["output_dtype"] = dtype
            summary[m_key]["elements"] = np.prod(
                summary[m_key]["output_shape"][1:]) / 1024.0 ** 2
            summary[m_key]["elements_mb"] = summary[m_key]["elements"] * elements_byte
            params = 0
            for _, param in module.named_parameters():
                params += param.data.nelement()
            summary[m_key]["nb_params"] = float(params)
            summary[m_key]["children_num"] = len(list(module.children()))

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)
                and not (module == model)):
            hooks.append(module.register_forward_hook(hook))

    summary = OrderedDict()
    hooks = []

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    x = [torch.rand(1, *in_size).type(torch.FloatTensor) for in_size in input_size]
    model.apply(register_hook)
    model(*x)

    for h in hooks:
        h.remove()

    data = []
    total_output = 0
    total_params = 0
    total_params_size = 0
    for _, param in model.named_parameters():
        dtype = str(param.data.dtype)
        param_bype = float(''.join([s for s in dtype if s.isdigit()])) / 8

        total_params += param.data.nelement()
        total_params_size += param.data.nelement() * param_bype
    total_params = total_params / 1024.0 ** 2
    total_params_size = total_params_size / 1024.0 ** 2

    def warp(*args, lchar='', rchar=''):
        out = []
        for arg in args:
            out.append('{}{}{}'.format(lchar, arg, rchar))
        return out

    for layer in summary:
        input_shape = str(summary[layer]["input_shape"])
        output_shape = str(summary[layer]["output_shape"])
        elements = '{:.02f}'.format(summary[layer]["elements"])
        elements_mb = '{:.02f}'.format(summary[layer]["elements_mb"])
        output_dtype = summary[layer]["output_dtype"]
        params = "{:.02f}".format(float(summary[layer]["nb_params"]) / 1024.0 ** 2)

        if summary[layer]["children_num"] == 0:
            total_output += summary[layer]["elements_mb"]
            head = [layer]
            line = warp(input_shape, output_shape, elements, elements_mb, output_dtype, params,
                        lchar=' ', rchar=' ')
        else:
            head = warp(layer, lchar='<', rchar='>')
            line = warp(input_shape, output_shape, elements, elements_mb, output_dtype, params,
                        lchar='<', rchar='> ')
        head.extend(line)
        data.append(head)

    table = tabulate(data, headers=headers, stralign='center')

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * 4. / (1024 ** 2.))
    total_output_size = 2. * total_output  # x2 for gradients
    total_size = total_params_size + total_output_size + total_input_size

    summary_msg = colored(
        "\nAll parameters (elements): {:.02f}M".format(total_params) +
        "\nInput size: {:.02f}MB".format(total_input_size) +
        "\nForward/backward pass size: {:.02f}MB".format(total_output_size) +
        "\nParams size: {:.02f}MB".format(total_params_size) +
        "\nEstimated Total Size: {:.02f}MB\n".format(total_size), 'cyan')

    logger.info(colored("List of All Features: \n", 'cyan') + table + summary_msg)


def _iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)


def register_hooks(var, id_dict):
    fn_dict = {}

    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input

        fn.register_hook(register_grad)

    _iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        try:
            grad_output = grad_output.data
        except:
            return False
        return grad_output.ne(grad_output).any() or grad_output.gt(1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='12',
                         ranksep='0.1',
                         height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="32,32"))

        def size_to_str(size):
            return '(' + ', '.join(map(str, size)) + ')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):
                u = fn.variable
                assert id(u) in id_dict
                node_name = id_dict[id(u)] + '\n' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                if fn not in fn_dict:
                    print("not in dict: ", fn)
                    return
                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(id(fn)), str(next_id))
        _iter_graph(var.grad_fn, build_graph)
        return dot
    return make_dot
