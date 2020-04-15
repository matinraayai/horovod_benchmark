"""
================================================================================
Author: Matin Raayai Ardakani
Distributed Training Tensorflow benchmark with MirroredStrategy.
Based on the tutorial here:
https://www.tensorflow.org/tutorials/distribute/custom_training
================================================================================
"""
from __future__ import absolute_import, division, print_function

import timeit
import argparse
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import applications
import numpy as np
# Benchmark settings:==========================================================#
parser = argparse.ArgumentParser(description='Distributed TensorFlow 2 Benchmark'
                                 'with MirroredStrategy',
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
# Strategy Setup:==============================================================#
strategy = tf.distribute.MirroredStrategy()
print('Number of devices in the strategy: {}'.format(strategy.num_replicas_in_sync))

# CIFAR100 Distributed Dataset Setup:==========================================#
tfds.disable_progress_bar()

datasets, info = tfds.load(name='cifar100', with_info=True, as_supervised=True)
cifar100_train, cifar100_test = datasets['train'], datasets['test']

batch_size = args.batch_size_per_replica * strategy.num_replicas_in_sync

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
                              .cache()\
                              .shuffle(args.buffer)\
                              .batch(batch_size)

# Distributed Dataset Creation:================================================#
train_dataset = strategy.experimental_distribute_dataset(train_dataset)

# Distributed Model Setup:=====================================================#
with strategy.scope():
    # Model Setup:=============================================================#
    model = applications.vgg16.VGG16(weights=None, pooling='max', classes=100)

    # Optimizer Setup:=========================================================#
    opt = tf.optimizers.Adam()

    # Distributed Loss Setup:==================================================#
    loss_object = tf.losses\
                    .CategoricalCrossentropy(from_logits=True,
                                             reduction=tf.losses.Reduction.NONE)
    def distributed_loss(y_label, predict):
        # The loss needs to be divided by the global batch size.
        return tf.nn.compute_average_loss(loss_object(y_label, predict),
                                          global_batch_size=batch_size)

    """
    A single benchmark step to be executed on each replica.
    @param dataset_inputs (x_input, y_label) tuple
    @return loss of the step in a single replica
    """
    def benchmark_step(dataset_inputs):
        x_input, y_label = dataset_inputs
        y_label = tf.reshape(y_label, (y_label.shape[0], 1))
        with tf.GradientTape() as tape:
            prediction = model(x_input, training=True)
            loss = distributed_loss(y_label, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    """
    Distributed benchmark step that calculates the reduced loss. 
    @param dataset_inputs (x_input, y_label) tuple
    @param loss of the step reduced
    """
    @tf.function
    def distributed_benchmark_step(dataset_inputs):
        replica_loss = strategy.experimental_run_v2(benchmark_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, replica_loss, axis=None)

    # Start of the benchmark logic:============================================#
    
    # Benchmark Warm-up:=======================================================#
    print('Running warmup...')
    train_dataset_iter = iter(train_dataset)
    for _ in range(args.num_warmup_batches):
        timeit.timeit(lambda: distributed_benchmark_step(next(train_dataset_iter)),
                      number=1)
    print('Warm-up finished.')
    # Actual Benchmark Start:==================================================#
    print('Running benchmark...')
    img_secs = []
    for epoch in range(args.num_epochs):
        time = 0.
        num_images = 0
        for dataset_inputs in train_dataset:
            time += timeit.timeit(lambda: distributed_benchmark_step(dataset_inputs),
                                  number=1)
            num_images += args.batch_size_per_replica
            print('Current forward pass speed per device: %.3f img/sec.' 
                  % (num_images / time))
            img_sec = num_images / time
            img_secs.append(img_sec)
        print('Epoch #%d: %.3f img/sec per device.' % (epoch, tf.reduce_mean(img_sec).numpy()))
        
    # Final results:===========================================================#
    img_sec_mean = tf.reduce_mean(img_secs)
    img_sec_conf = 1.96 * tf.math.reduce_std(img_secs)
    print('Img/sec per device: %.3f +-%.3f' % (img_sec_mean, img_sec_conf))
    print('Total img/sec on %d Device(s): %.3f +-%.3f' %
          (strategy.num_replicas_in_sync, strategy.num_replicas_in_sync * img_sec_mean,
           strategy.num_replicas_in_sync * img_sec_conf))
