import numpy as np
import tensorflow as tf

from keras.datasets import mnist


inputs = tf.placeholder(shape=(None, 28, 28, 1), dtype=tf.float32)
labels = tf.placeholder(shape=(None), dtype=tf.int32)
p = tf.placeholder(shape=(), dtype=tf.float32)

conv1 = tf.layers.conv2d( inputs=inputs,
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

dropout = tf.nn.dropout(dense, keep_prob=1-p)

logits = tf.layers.dense(inputs=dropout, units=10)

predictions = {
    "classes": tf.argmax(logits, axis=-1),
    "probabilities": tf.nn.softmax(logits, name="softmax")
}

accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

saver = tf.train.Saver()


def generate_batches(X, y, batch_size):
    for i in range(len(X)//batch_size + 1):
        start = i * batch_size
        end = (i+1) * batch_size
        yield (X[start:end], y[start:end])

def get_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    num_val_samples = 10000
    X_val = X_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]

    X_train = X_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

with tf.Session() as sess:
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data()
    print("train shapes: ", X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    dropout_p = 0.1
    num_epochs = 10
    batch_size = 1024
   
    train_model = False

    checkpoint_path = "models/tf_test_cnn.ckpt"

    if train_model:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for epoch in range(num_epochs):
            losses = []
            for (X, y) in generate_batches(X_train, y_train, batch_size): 
                loss_, _ = sess.run([loss, train_op], feed_dict={
                    inputs: X,
                    labels: y,
                    p: dropout_p
                })
                losses.append(loss_)
            losses = np.concatenate(losses)
            average_loss = np.mean(losses)
                

            if epoch % 5 == 0:
                predictions_ = []
                for (X,y) in generate_batches(X_val, y_val, batch_size):
                    batch_preds_ = sess.run(predictions["classes"], feed_dict={
                        inputs: X,
                        labels: y,
                        p: 0
                    })
                    predictions_.append(batch_preds_)
                predictions_ = np.concatenate(predictions_)
                acc = len(np.argwhere(predictions_ == y_val))/len(y_val)
                print("Epoch: {}\tLoss: {}\tVal acc: {}".format(epoch, average_loss, acc))
            else:
                print("Epoch: {}\tLoss: {}".format(epoch, average_loss))
        
        saver.save(sess, checkpoint_path)
        print("Saved model")
    else:
        saver.restore(sess, checkpoint_path)
        print("Model restored")

    predictions_ = []
    for (X,y) in generate_batches(X_test, y_test, batch_size):
        batch_preds_ = sess.run(predictions["classes"], feed_dict={
            inputs: X,
            labels: y,
            p: 0
        })
        predictions_.append(batch_preds_)
    predictions_ = np.concatenate(predictions_)
    acc = len(np.argwhere(predictions_ == y_test))/len(y_test)
    print("Test acc: {}".format(acc))
       
    wrong_index = np.argwhere(predictions_ != y_test)[0, 0]

    print("Test image index: {}".format(wrong_index))
    
    T = 3
    img = [X_test[wrong_index]] * T
    img_label = [y_test[wrong_index]] * T
    print("Assigned label: {}".format(img_label))
    print("Correct label: {}".format(y_test[wrong_index]))

    preds_ = sess.run(predictions, feed_dict={inputs:img, labels:img_label, p:0.2})
    probabilities = preds_['probabilities']
    print(probabilities.shape)
    pred_means = np.mean(probabilities, axis=0)
    pred_std = np.std(probabilities, axis=0)
    
    print("means: ", pred_means)
    print("stds: ", pred_std)

