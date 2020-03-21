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
import Functions
import model_setting

from tensorflow.python.platform import gfile

data_dir = 'F:\\实习\\Finaltask\\code\\inputdata'
validation_percentage = 0.1
testing_percentage = 0.1
training_percentage = 1 - validation_percentage - testing_percentage
summaries_dir = 'F:\\实习\\Finaltask\\code\\tmp'
train_dir = 'F:\\实习\\Finaltask\\code\\tmp\\speech_commands_train'
time_shift_ms = 100.0
sample_rate = 16000
how_many_training_steps = [1500, 300]
learning_rate = [0.001, 0.0001]
is_training = True
batch_size = 100
wanted_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
String2int = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
              'nine': 9}

tf.logging.set_verbosity(tf.logging.INFO)
sess = tf.InteractiveSession()

model_settings = model_setting.prepare_model_settings(len(wanted_words), sample_rate, 1000, 30.0, 10.0, 40)
# print(model_settings)
"""
    将数据分到不同的部分('validation', 'testing', 'training')
"""

data_index = {'validation': [], 'testing': [], 'training': []}
search_path = os.path.join(data_dir, '*', '*.wav')
for wav_path in gfile.Glob(search_path):
    _, word = os.path.split(os.path.dirname(wav_path))
    set_index = Functions.set_path(wav_path, validation_percentage, testing_percentage)
    data_index[set_index].append({'label': word, 'file': wav_path})
print('validation:' + str(len(data_index['validation'])))
print('testing:' + str(len(data_index['testing'])))
print('training:' + str(len(data_index['training'])))
# print(data_index['testing'])
for set_index in ['validation', 'testing', 'training']:
    random.shuffle(data_index[set_index])
# print(data_index['testing'])

"""
    创建TensorFlow图
"""

with tf.name_scope('data'):
    desired_samples = model_settings['desired_samples']
    # print(desired_samples.name)
    wav_filename_placeholder_ = tf.placeholder(tf.string, [], name='wav_filename')
    # print(wav_filename_placeholder_.name)
    wav_loader = io_ops.read_file(wav_filename_placeholder_)
    # print(wav_loader.shape)
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1, desired_samples=desired_samples)
    foreground_volume_placeholder_ = tf.placeholder(tf.float32, [], name='foreground_volume')
    scaled_foreground = tf.multiply(wav_decoder.audio, foreground_volume_placeholder_)
    time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2], name='time_shift_padding')
    time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2], name='time_shift_offset')
    padded_foreground = tf.pad(scaled_foreground, time_shift_padding_placeholder_, mode='CONSTANT')
    sliced_foreground = tf.slice(padded_foreground, time_shift_offset_placeholder_, [desired_samples, -1])
    spectrogram = contrib_audio.audio_spectrogram(sliced_foreground, window_size=model_settings['window_size_samples'],
                                                  stride=model_settings['window_stride_samples'],
                                                  magnitude_squared=True)
    tf.summary.image('spectrogram', tf.expand_dims(spectrogram, -1), max_outputs=1)
    output_ = contrib_audio.mfcc(spectrogram, wav_decoder.sample_rate,
                                 dct_coefficient_count=model_settings['fingerprint_width'])
    tf.summary.image('mfcc', tf.expand_dims(output_, -1), max_outputs=1)
    merged_summaries_ = tf.summary.merge_all(scope='data')
    summary_writer_ = tf.summary.FileWriter(summaries_dir + '/data', tf.get_default_graph())

fingerprint_size = model_settings['fingerprint_size']
label_count = model_settings['label_count']
time_shift_samples = int((time_shift_ms * sample_rate) / 1000)

input_placeholder = tf.placeholder(tf.float32, [None, fingerprint_size], name='fingerprint_input')
fingerprint_input = input_placeholder

"""
    创建create_conv_model
    
"""

if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
input_frequency_size = model_settings['fingerprint_width']
input_time_size = model_settings['spectrogram_length']
fingerprint_4d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size, 1])
first_filter_width = 8
first_filter_height = 20
first_filter_count = 64
first_weights = tf.get_variable(
    name='first_weights',
    initializer=tf.truncated_normal_initializer(stddev=0.01),
    shape=[first_filter_height, first_filter_width, 1, first_filter_count])
first_bias = tf.get_variable(
    name='first_bias',
    initializer=tf.zeros_initializer,
    shape=[first_filter_count])
first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                          'SAME') + first_bias
first_relu = tf.nn.relu(first_conv)
if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
else:
    first_dropout = first_relu
max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
second_filter_width = 4
second_filter_height = 10
second_filter_count = 64
second_weights = tf.get_variable(
    name='second_weights',
    initializer=tf.truncated_normal_initializer(stddev=0.01),
    shape=[
        second_filter_height, second_filter_width, first_filter_count,
        second_filter_count
    ])
second_bias = tf.get_variable(
    name='second_bias',
    initializer=tf.zeros_initializer,
    shape=[second_filter_count])
second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                           'SAME') + second_bias
second_relu = tf.nn.relu(second_conv)
if is_training:
    second_dropout = tf.nn.dropout(second_relu, dropout_prob)
else:
    second_dropout = second_relu
second_conv_shape = second_dropout.get_shape()
second_conv_output_width = second_conv_shape[2]
second_conv_output_height = second_conv_shape[1]
second_conv_element_count = int(
    second_conv_output_width * second_conv_output_height *
    second_filter_count)
flattened_second_conv = tf.reshape(second_dropout,
                                   [-1, second_conv_element_count])
label_count = model_settings['label_count']
final_fc_weights = tf.get_variable(
    name='final_fc_weights',
    initializer=tf.truncated_normal_initializer(stddev=0.01),
    shape=[second_conv_element_count, label_count])
final_fc_bias = tf.get_variable(
    name='final_fc_bias',
    initializer=tf.zeros_initializer,
    shape=[label_count])
final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias

ground_truth_input = tf.placeholder(tf.int64, [None], name='groundtruth_input')
with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(labels=ground_truth_input, logits=final_fc)
with tf.name_scope('train'):
    learning_rate_input = tf.placeholder(tf.float32, [], name='learning_rate_input')
    train_step = tf.train.GradientDescentOptimizer(learning_rate_input).minimize(cross_entropy_mean)
predicted_indices = tf.argmax(final_fc, 1)  # 返回最大值
correct_prediction = tf.equal(predicted_indices, ground_truth_input)  # 矩阵比较，相等返回true
confusion_matrix = tf.confusion_matrix(ground_truth_input, predicted_indices, num_classes=label_count)  # 混淆矩阵
evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.get_default_graph().name_scope('eval'):
    tf.summary.scalar('cross_entropy', cross_entropy_mean)
    tf.summary.scalar('accuracy', evaluation_step)

global_step = tf.train.get_or_create_global_step()
increment_global_step = tf.assign(global_step, global_step + 1)

saver = tf.train.Saver(tf.global_variables())

merged_summaries = tf.summary.merge_all(scope='eval')
train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
validation_writer = tf.summary.FileWriter(summaries_dir + '/validation')

tf.global_variables_initializer().run()

start_step = 1

tf.logging.info('Training from step: %d ', start_step)
tf.train.write_graph(sess.graph_def, train_dir, 'conv' + '.pbtxt')

with gfile.GFile(
        os.path.join(train_dir, 'conv' + '_labels.txt'),
        'w') as f:
    f.write('\n'.join(wanted_words))

training_steps_max = np.sum(how_many_training_steps)
for training_step in xrange(start_step, training_steps_max + 1):
    training_steps_sum = 0
    for i in range(len(how_many_training_steps)):
        training_steps_sum += how_many_training_steps[i]
        if training_step <= training_steps_sum:
            learning_rate_value = learning_rate[i]
            break
    mode = 'training'
    offset = 0
    candidates = data_index['training']
    sample_count = max(0, min(100, len(candidates) - offset))
    data = np.zeros((sample_count, model_settings['fingerprint_size']))
    labels = np.zeros(sample_count)
    pick_deterministically = (mode != 'training')
    desired_samples = model_settings['desired_samples']
    for i in xrange(offset, offset + sample_count):
        sample_index = np.random.randint(len(candidates))
        sample = candidates[sample_index]
        time_shift = time_shift_samples
        if time_shift > 0:
            time_shift_amount = np.random.randint(-time_shift, time_shift)
        else:
            time_shift_amount = 0
        if time_shift_amount > 0:
            time_shift_padding = [[time_shift_amount, 0], [0, 0]]
            time_shift_offset = [0, 0]
        else:
            time_shift_padding = [[0, -time_shift_amount], [0, 0]]
            time_shift_offset = [-time_shift_amount, 0]
        input_dict = {
            wav_filename_placeholder_: sample['file'],
            time_shift_padding_placeholder_: time_shift_padding,
            time_shift_offset_placeholder_: time_shift_offset,
        }
        input_dict[foreground_volume_placeholder_] = 1
        summary, data_tensor = sess.run([merged_summaries_, output_], feed_dict=input_dict)
        summary_writer_.add_summary(summary)
        data[i - offset, :] = data_tensor.flatten()
        label_index = int(String2int[sample['label']])
        labels[i - offset] = label_index
    train_fingerprints = data
    train_ground_truth = labels
    train_summary, train_accuracy, cross_entropy_value, _, _, predit = sess.run(
        [
            merged_summaries,
            evaluation_step,
            cross_entropy_mean,
            train_step,
            increment_global_step,
            predicted_indices
        ],
        feed_dict={
            fingerprint_input: train_fingerprints,
            ground_truth_input: train_ground_truth,
            learning_rate_input: learning_rate_value,
            dropout_prob: 0.5
        })
    train_writer.add_summary(train_summary, training_step)
    tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                    (training_step, learning_rate_value, train_accuracy * 100,
                     cross_entropy_value))
    # tf.logging.info('%s,%s' % (predit, train_ground_truth))
    is_last_step = (training_step == training_steps_max)
    if (training_step % 100) == 0 or is_last_step:
        set_size = len(data_index['validation'])
        total_accuracy = 0
        total_conf_matrix = None
        for i in xrange(0, set_size, 200):
            mode = 'validation'
            candidates = data_index['validation']
            sample_count = max(0, min(200, len(candidates) - i))
            data = np.zeros((sample_count, model_settings['fingerprint_size']))
            labels = np.zeros(sample_count)
            desired_samples = model_settings['desired_samples']
            pick_deterministically = (mode != 'training')
            for j in xrange(i, i + sample_count):
                sample_index = j
                sample = candidates[sample_index]
                time_shift_amount = 0
                time_shift_padding = [[0, -time_shift_amount], [0, 0]]
                time_shift_offset = [-time_shift_amount, 0]
                input_dict = {
                    wav_filename_placeholder_: sample['file'],
                    time_shift_padding_placeholder_: time_shift_padding,
                    time_shift_offset_placeholder_: time_shift_offset,
                }
                input_dict[foreground_volume_placeholder_] = 1
                summary, data_tensor = sess.run([merged_summaries_, output_], feed_dict=input_dict)
                summary_writer_.add_summary(summary)
                data[j - i, :] = data_tensor.flatten()
                label_index = String2int[sample['label']]
                labels[j - i] = label_index
            validation_fingerprints = data
            validation_ground_truth = labels
            validation_summary, validation_accuracy, conf_matrix = sess.run(
                [merged_summaries, evaluation_step, confusion_matrix],
                feed_dict={
                    fingerprint_input: validation_fingerprints,
                    ground_truth_input: validation_ground_truth,
                    dropout_prob: 1.0
                })
            validation_writer.add_summary(validation_summary, training_step)
            batch_size_1 = min(batch_size, set_size - i)
            total_accuracy += (validation_accuracy * batch_size_1) / set_size
            if total_conf_matrix is None:
                total_conf_matrix = conf_matrix
            else:
                total_conf_matrix += conf_matrix
        tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
        tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                        (training_step, total_accuracy * 100, set_size))

    if (training_step % 100 == 0 or
            training_step == training_steps_max):
        checkpoint_path = os.path.join(train_dir,
                                       'conv' + '.ckpt')
        tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
        saver.save(sess, checkpoint_path, global_step=training_step)

set_size = len(data_index['testing'])
tf.logging.info('set_size=%d', set_size)
total_accuracy = 0
total_conf_matrix = None
for i in xrange(0, set_size, 100):
    mode = 'testing'
    candidates = data_index['testing']
    sample_count = max(0, min(100, len(candidates) - i))
    data = np.zeros((sample_count, model_settings['fingerprint_size']))
    labels = np.zeros(sample_count)
    desired_samples = model_settings['desired_samples']
    pick_deterministically = (mode != 'training')
    for j in xrange(i, i + sample_count):
        sample_index = j
        sample = candidates[sample_index]
        time_shift_amount = 0
        time_shift_padding = [[0, -time_shift_amount], [0, 0]]
        time_shift_offset = [-time_shift_amount, 0]
        input_dict = {
            wav_filename_placeholder_: sample['file'],
            time_shift_padding_placeholder_: time_shift_padding,
            time_shift_offset_placeholder_: time_shift_offset,
        }
        input_dict[foreground_volume_placeholder_] = 1
        summary, data_tensor = sess.run([merged_summaries_, output_], feed_dict=input_dict)
        summary_writer_.add_summary(summary)
        data[j - i, :] = data_tensor.flatten()
        label_index = String2int[sample['label']]
        labels[j - i] = label_index
    test_fingerprints = data
    test_ground_truth = labels
    test_accuracy, conf_matrix = sess.run(
        [evaluation_step, confusion_matrix],
        feed_dict={
            fingerprint_input: test_fingerprints,
            ground_truth_input: test_ground_truth,
            dropout_prob: 1.0
        })
    batch_size = min(100, set_size - i)
    total_accuracy += (test_accuracy * batch_size) / set_size
    if total_conf_matrix is None:
        total_conf_matrix = conf_matrix
    else:
        total_conf_matrix += conf_matrix
tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100,
                                                         set_size))
