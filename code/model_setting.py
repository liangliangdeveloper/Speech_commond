from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import math
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

"""
    对于训练模型的设置与选择
"""


def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms, feature_bin_count,
                           ):
    """

    :param label_count:有多少类要被组织
    :param sample_rate:每秒音频采样数
    :param clip_duration_ms:音频长度（ms）
    :param window_size_ms:窗口长度
    :param window_stride_ms:窗口移动长度
    :param feature_bin_count:用于分析的频率区数量
    :return:
    """

    desired_samples = int(sample_rate * clip_duration_ms / 1000)               # 每秒采样数*声音长度（毫秒）/1000 = 声音采样数
    window_size_samples = int(sample_rate * window_size_ms / 1000)             # 每秒采样数*频率分析窗口长度（毫秒）/1000 = 窗口分析采样数
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)         # 每秒采样数*窗口移动的秒数（毫秒）/1000 = 窗口移动长度
    length_minus_window = (desired_samples - window_size_samples)              # 采样数-窗口采样数
    if length_minus_window < 0:
        spectrogram_length = 0                                                 # 光谱图长度
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)   # 我可以产生多少个光谱图，1是自己，然后后面int一下

        average_window_width = -1
        fingerprint_width = feature_bin_count

    fingerprint_size = fingerprint_width * spectrogram_length                   #频率,宽度*光谱图个数
    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'fingerprint_width': fingerprint_width,
        'fingerprint_size': fingerprint_size,
        'label_count': label_count,
        'sample_rate': sample_rate,
        'average_window_width': average_window_width
    }
