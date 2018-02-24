import numpy as np
import tensorflow as tf

from keras.datasets import mnist


inputs = tf.placeholder(shape=(None, 28, 28, 1), dtype=tf.float32)
labels = tf.placeholder(shape=(None), dtype=tf.int32)
p = tf.placeholder(shape=(), dtype=tf.float32)

conv1 = tf.layers.conv2d(
    inputs=inputs,
    filters=32,
    kernel_size=[5,5],
    padding="same",
    activation=tf.nn.relu)

pool1 = tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=[2,2],
    strides=2)

conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5,5],
    padding="same",
    activation=tf.nn.relu)

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

pool2_flat = tf.reshape(pool2, [-1, 7*7*64])

dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

dropout = tf.layers.dropout(
    inputs=dense,
    rate=1-p
)

logits = tf.layers.dense(inputs=dropout, units=10)

predictions = {
    "classes": tf.argmax(logits, axis=-1),
    "probabilities": tf.nn.softmax(logits, name="softmax")
}

accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

with tf.Session() as sess:
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    sess.run(tf.global_variables_initializer())

    dropout_p = 0.1

    accuracy_ = sess.run(accuracy, feed_dict={
        inputs: X_train,
        labels: y_train,
        p: dropout_p
    })

    print(accuracy_)