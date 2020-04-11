"""
# Modified the Horovod benchmark for TensorFlow Distributed.
# This benchmark only runs vgg-16 now.
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
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import applications

# Benchmark settings
parser = argparse.ArgumentParser(description='TensorFlow 2 Distributed Synthetic Benchmark'
                                             'for VGG-16',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')

parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')

parser.add_argument('--num-iters', type=int, default=100,
                    help='number of benchmark iterations')

args = parser.parse_args()


# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')


# Set up standard model. 
model = applications.vgg16.VGG16(weights=None, pooling='max', classes=1000)

opt = tf.optimizers.SGD(0.01)

data = tf.random.uniform([args.batch_size, 224, 224, 3])
target = tf.random.uniform([args.batch_size, 1], minval=0, maxval=999, dtype=tf.int64)


@tf.function
def benchmark_step():

    # Horovod: use DistributedGradientTape
    with tf.GradientTape() as tape:
        probs = model(data, training=True)
        loss = tf.losses.categorical_crossentropy(target, probs)

    # Horovod: add Horovod Distributed GradientTape.

    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))


print('Batch size: %d' % args.batch_size)
device = 'GPU'
physical_devices = tf.config.list_physical_devices('GPU') 
print("Num GPUs:", len(physical_devices))

with tf.device(device):
    # Warm-up
    print('Running warmup...')
    timeit.timeit(lambda: benchmark_step(),
                  number=args.num_warmup_batches)

    # Benchmark
    print('Running benchmark...')
    img_secs = []
    for x in range(args.num_iters):
        time = timeit.timeit(lambda: benchmark_step(),
                             number=args.num_batches_per_iter)
        img_sec = args.batch_size * args.num_batches_per_iter / time
        print('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
        img_secs.append(img_sec)

    # Results
    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    print('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
    print('Total img/sec on %d %s(s): %.1f +-%.1f' %
        (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))
