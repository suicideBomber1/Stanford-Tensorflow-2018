# Logistic Regression

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


learning_rate = 0.01
batch_size = 128
n_epochs = 30

# Create placeholders for the model
# MNIST data is of shape 28x28
# Therfore each image is a tensor of shape 1x784

X = tf.placeholder(tf.float32, shape=[batch_size, 784], name='X_placeholder')
Y = tf.placeholder(tf.float32, shape=[batch_size, 10], name='Y_placeholder')

# Creating weights and biases

w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name='weights')
b = tf.Variable(tf.zeros([1, 10]), name='biases')

# Building the model

logits = tf.matmul(X, w) + b

# Defining the loss function

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy)

# Creating the optimizer

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Training and running our model

with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graph/logistic_reg', sess.graph)
    n_batches = int(mnist.train.num_examples / batch_size)
    for i in range(n_epochs):
        total_loss = 0
        for _ in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
            total_loss += loss_batch
        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))

    print("Total time : {0}".format(time.time() - start_time))
    print("Optimizstion finished!")

    n_batches = int(mnist.test.num_examples / batch_size)
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        _, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict={X: X_batch, Y: Y_batch})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += sess.run(accuracy)

print("Accuracy: {0}".format(total_correct_preds / mnist.test.num_examples))
writer.close()
