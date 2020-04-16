"""
# Single GPU VGG-16 Benchmark for Tensorflow.
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
from __future__ import absolute_import, division, print_function

import timeit
import argparse
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import applications

# Benchmark settings:==========================================================#
parser = argparse.ArgumentParser(description='TensorFlow 2 single GPU benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-epochs', type=int, default=10,
                    help='number of epochs for the benchmark')
parser.add_argument('--num-batches-per-epoch', type=int, default=10,
                    help='number of batches per epoch')
parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size per a single GPU (replica)')
parser.add_argument('--buffer', type=int, default=10000,
                    help='buffer size for the distributed training dataset')
parser.add_argument('--num-warmup-batches', type=int, default=3,
                    help='number of warm-up batches that don\'t count towards the benchmark')
args = parser.parse_args()

# CIFAR100 Dataset Setup:======================================================#
tfds.disable_progress_bar()

datasets, info = tfds.load(name='cifar100', with_info=True, as_supervised=True)
cifar100_train, cifar100_test = datasets['train'], datasets['test']


'''
Pre-processing the CIFAR100 Dataset to be fed to the VGG-16 network.
'''
def pre_process(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32)
    image /= 255
    label = tf.cast(label, tf.float32)
    return image, label

train_dataset = cifar100_train.map(pre_process)\
                              .shuffle(args.buffer)\
                              .batch(args.batch_size)

# Standard model:==============================================================#
model = applications.vgg16.VGG16(weights=None, pooling='max', classes=100)

# Optimizer Setup:=============================================================#
opt = tf.optimizers.Adam()

"""
A single benchmark step to be executed on each replica.
@param dataset_inputs (x_input, y_label) tuple
@param first_batch True if this is the first batch of the training, 
False otherwise
"""
@tf.function
def benchmark_step(dataset_inputs):
    x_input, y_label = dataset_inputs
    y_label = tf.reshape(y_label, (y_label[0], 1))
    with tf.GradientTape() as tape:
        prediction = model(x_input, training=True)
        loss = tf.losses.categorical_crossentropy(y_label, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    
# Start of the benchmark logic:================================================#
with tf.device('GPU'):
    # Benchmark Warm-up:=======================================================#
    print('Running warmup...')
    # Horovod initial Batch:===================================================#
    train_dataset_iter = iter(train_dataset)
    for _ in range(args.num_warmup_batches):
        timeit.timeit(lambda: benchmark_step(next(train_dataset_iter)),
                      number=1)
    print('Warmup finished.')
    # Actual Benchmark Start:==================================================#
    print('Running benchmark...')
    img_secs = []
    for epoch in range(args.num_epochs):
        for dataset_inputs in train_dataset:
            time = timeit.timeit(lambda: benchmark_step(dataset_inputs),
                                  number=1)
            num_images = len(dataset_inputs[0])
            img_sec = num_images / time 
            print('Current forward pass speed: %.3f img/sec.' 
                  % img_sec)
            img_secs.append(img_sec)
        print('Epoch #%d: %.3f img/sec' % (epoch, tf.reduce_min(img_sec).numpy()))

    # Final results:===========================================================#
    img_sec_mean = tf.reduce_min(img_secs)
    img_sec_conf = 1.96 * tf.math.reduce_std(img_secs)
    print('Img/sec: %.3f +-%.3f' % (img_sec_mean.numpy(),
                                               img_sec_conf.numpy()))
