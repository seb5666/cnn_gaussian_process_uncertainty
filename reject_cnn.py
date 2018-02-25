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

reject_neuron = tf.layers.dense(inputs=dropout, units=1)

predictions = {
    "classes": tf.argmax(logits, axis=-1),
    "probabilities": tf.nn.softmax(logits, name="softmax"),
    "reject_p": tf.sigmoid(reject_neuron)
}

accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])

#loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
labels_one_hot = tf.one_hot(labels, 10)
log_probs = tf.log(predictions['probabilities'] + 1e-10)
cross_entropy_A = tf.multiply(labels_one_hot, log_probs)
cross_entropy_B = tf.multiply(labels_one_hot, tf.log(1-predictions['reject_p']+1e-10) + log_probs) + tf.multiply(1-labels_one_hot, tf.log(predictions['reject_p'] + 1e-10))
cross_entropy = (1 - predictions['reject_p']) * cross_entropy_A + predictions['reject_p'] * cross_entropy_B
loss = - tf.reduce_sum(cross_entropy, axis=-1)

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
    num_epochs = 20
    batch_size = 1024
   
    train_model = True

    checkpoint_path = "models/tf_reject_cnn.ckpt"

    if train_model:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for epoch in range(num_epochs):
            losses = []
            for (X, y) in generate_batches(X_train, y_train, batch_size): 
                log_probs_, ls_, cpA_, cpB_, cp_, loss_, _ = sess.run([log_probs, labels_one_hot, cross_entropy_A, cross_entropy_B, cross_entropy, loss, train_op], feed_dict={
                    inputs: X,
                    labels: y,
                    p: dropout_p
                })
                #print(log_probs_)
                #print(ls_)
                #print(cpA_)
                #print(cpB_)
                #print(cp_)
                #print(loss_.shape)
                #print(loss_)
                #print(loss_.shape)
                #exit()
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
    
    probabilities_ = []
    predictions_ = []
    reject_ps_ = []
    for (X,y) in generate_batches(X_test, y_test, batch_size):
        batch_probs_, batch_preds_, batch_reject_ps_ = sess.run([predictions["probabilities"], predictions["classes"], predictions["reject_p"]], feed_dict={
            inputs: X,
            labels: y,
            p: 0
        })
        predictions_.append(batch_preds_)
        reject_ps_.append(batch_reject_ps_)
        probabilities_.append(batch_probs_)
    predictions_ = np.concatenate(predictions_)
    reject_ps_ = np.concatenate(reject_ps_)[:,0]
    probabilities_ = np.concatenate(probabilities_)

    print(probabilities_.shape)
    print(predictions_.shape)
    print(reject_ps_.shape)
    acc = len(np.argwhere(predictions_ == y_test))/len(y_test)

    print("Test acc: {}".format(acc))

    rejected_imgs = np.argwhere(np.argmax(probabilities_, axis=-1) < reject_ps_)

    print("Num rejected: {}".format(len(rejected_imgs)))
