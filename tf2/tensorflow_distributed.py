"""
Distributed Training in Tensorflow with Multiworker Strategy.
"""
from __future__ import absolute_import, division, print_function

import timeit
import os
import argparse
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import applications
import numpy as np
# Benchmark settings
parser = argparse.ArgumentParser(description='Distributed TensorFlow 2 Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-epochs', type=int, default=10,
                    help='input batch size')
parser.add_argument('--num-batches-per-epoch', type=int, default=10,
                    help='number of batches per epoch') 
parser.add_argument('--batch-size-per-replica', type=int, default=64,
                    help='input batch size per a single GPU.')
parser.add_argument('--buffer', type=int, default=10000,
                    help='Buffer size for the input dataset.')
parser.add_argument('--num-warmup-batches', type=int, default=3,
                    help='number of warm-up batches that don\'t count towards benchmark')
args = parser.parse_args()

# Strategy Setup:
strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


# Dataset Setup:

tfds.disable_progress_bar()

datasets, info = tfds.load(name='cifar10', with_info=True, as_supervised=True)

mnist_train, mnist_test = datasets['train'], datasets['test']

# You can also do info.splits.total_num_examples to get the total
# number of examples in the dataset.

num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

batch_size = args.batch_size_per_replica * strategy.num_replicas_in_sync


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


train_dataset = mnist_train.map(scale).cache().shuffle(args.buffer).batch(batch_size)
eval_dataset = mnist_test.map(scale).batch(batch_size)

# Standard Model Setup:
with strategy.scope():
    model = applications.vgg16.VGG16(weights=None, pooling='max', classes=1000)

opt = tf.optimizers.Adam()

@tf.function
def benchmark_step(x_input, y_label):
    with tf.GradientTape() as tape:
        prediction = model(x_input, training=True)
        loss = tf.losses.categorical_crossentropy(y_label, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

# Warm-up
print('Running warmup...')
for x, y in train_dataset.take(args.num_warmup_batches):
    timeit.timeit(lambda: benchmark_step(x, y), number=1)

# Benchmark
print('Running benchmark...')
img_secs = []
for epoch in range(args.num_epochs):
    time = 0.
    for x, y in train_dataset.take(args.num_batches_per_epoch):
        time += timeit.timeit(lambda: benchmark_step(x, y),
                              number=args.num_batches_per_iter)
    img_sec = args.num_batches_per_epoch / time
    print('Epoch #%d: %.1f img/sec per device.' % (i, img_sec))
    img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
print('Img/sec per device: %.1f +-%.1f' % (img_sec_mean, img_sec_conf))
print('Total img/sec on %d Device(s): %.1f +-%.1f' %
      (strategy.num_replicas_in_sync, strategy.num_replicas_in_sync * img_sec_mean,
       strategy.num_replicas_in_sync * img_sec_conf))
