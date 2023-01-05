import os
import math
import torch
import logging
import yaml

def overlap_and_add(signal, frame_step):
    """
    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length
    subframe_signal = signal.reshape(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long() # signal may in GPU or CPU
    if signal.device.type == "cuda":
        frame = frame.cuda()
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result

def remove_pad(inputs, inputs_lengths):
    """
    Args:
        inputs: torch.Tensor, [B, C, T] or [B, T], B is batch size
        inputs_lengths: torch.Tensor, [B]
    Returns:
        results: a list containing B items, each item is [C, T], T varies
    """
    results = []
    dim = inputs.dim()
    if dim == 3:
        C = inputs.size(1)
    for input, length in zip(inputs, inputs_lengths):
        if dim == 3: # [B, C, T]
            results.append(input[:,:length].view(C, -1).cpu().numpy())
        elif dim == 2:  # [B, T]
            results.append(input[:length].view(-1).cpu().numpy())
    return results

def get_logger(name, format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
               date_format='%Y-%m-%d %H:%M:%S', file=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # file or console
    handler = logging.StreamHandler() if not file else logging.FileHandler(
        name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = get_logger(__name__)

def parse(opt_path, is_tain=True):
    '''
       opt_path: the path of yml file
       is_train: True
    '''
    logger.info('Reading .yml file .......')
    with open(opt_path,mode='r') as f:
        opt = yaml.load(f,Loader=yaml.FullLoader)
    # Export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    #os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    #logger.info('Export CUDA_VISIBLE_DEVICES = {}'.format(gpu_list))

    # is_train into option
    opt['is_train'] = is_tain

    return opt

def setLogger(log_path):
    """
    Define log file
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        #Logging to file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        #Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def save_opt(opt, opt_path):
    '''
       opt_path: the path of yml file
       is_train: True
    '''
    logger.info('Saving .yml file .......')
    with open(opt_path,mode='w') as f:
        yaml.dump(opt, f)
