"""
================================================================================
# Horovod Synthetic benchmark, modified to take in real data to be trained
# with VGG-16 on CIFAR100.
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved
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
import tensorflow as tf
import tensorflow_datasets as tfds
import horovod.tensorflow as hvd
from tensorflow.keras import applications


# Logging function on Rank 0:==================================================#
def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')

# Benchmark settings:==========================================================#
parser = argparse.ArgumentParser(description='TensorFlow 2 on Horovod Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-epochs', type=int, default=10,
                    help='number of epochs for the benchmark')
parser.add_argument('--num-batches-per-epoch', type=int, default=10,
                    help='number of batches per epoch')
parser.add_argument('--batch-size-per-replica', type=int, default=64,
                    help='batch size per a single GPU (replica)')
parser.add_argument('--buffer', type=int, default=10000,
                    help='buffer size for the distributed training dataset')
parser.add_argument('--num-warmup-batches', type=int, default=3,
                    help='number of warm-up batches that don\'t count towards the benchmark')
args = parser.parse_args()

# Horovod Program Init:========================================================#
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process):=====#
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

log("Number of Horovod Processes (GPUs): {}".format(hvd.size()))
# CIFAR100 Dataset Setup:======================================================#
tfds.disable_progress_bar()

datasets, info = tfds.load(name='cifar100', with_info=True, as_supervised=True)
cifar100_train, cifar100_test = datasets['train'], datasets['test']

batch_size = args.batch_size_per_replica * hvd.size()

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
                              .batch(batch_size)

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
def benchmark_step(dataset_inputs, first_batch=False):
    x_input, y_label = dataset_inputs
    y_label = tf.reshape(y_label, (y_label[0], 1))
    with tf.GradientTape() as tape:
        prediction = model(x_input, training=True)
        loss = tf.losses.categorical_crossentropy(y_label, prediction)
    # Horovod: add Horovod Distributed GradientTape for reduction:=============#
    tape = hvd.DistributedGradientTape(tape)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    # Horovod: broadcast initial variable states from rank 0 to all other 
    # processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    # Note: broadcast should be done after the first gradient step to ensure 
    # optimizer initialization.
    if first_batch:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)

# Start of the benchmark logic:================================================#
with tf.device('GPU'):
    # Benchmark Warm-up:=======================================================#
    log('Running warmup...')
    # Horovod initial Batch:===================================================#
    train_dataset_iter = iter(train_dataset)
    benchmark_step(next(train_dataset_iter), first_batch=True)
    for _ in range(args.num_warmup_batches):
        timeit.timeit(lambda: benchmark_step(next(train_dataset_iter)),
                      number=1)
    log('Warmup finished.')
    # Actual Benchmark Start:==================================================#
    log('Running benchmark...')
    img_secs = []
    for epoch in range(args.num_epochs):
        time = 0.
        num_images = 0
        for dataset_inputs in train_dataset:
            time += timeit.timeit(lambda: benchmark_step(dataset_inputs),
                                  number=1)
            num_images += args.batch_size_per_replica
            print('Current forward pass speed per device: %.3f img/sec.' 
                  % (num_images / time))
            img_sec = num_images / time
            img_secs.append(img_sec)
        print('Epoch #%d: %.3f img/sec per device' % (epoch, img_sec))

    # Final results:===========================================================#
    img_sec_mean = tf.reduce_min(img_secs)
    img_sec_conf = 1.96 * tf.math.reduce_std(img_secs)
    print('Img/sec per device: %.3f +-%.3f' % (img_sec_mean.numpy(),
                                               img_sec_conf.numpy()))
    img_sec_mean = hvd.allreduce(img_sec_mean, op=1).numpy()
    img_sec_conf = hvd.allreduce(img_sec_conf, op=1).numpy()
    log('Total img/sec on %d Device(s): %.3f +-%.3f' %
        (hvd.size(), img_sec_mean, img_sec_conf))
