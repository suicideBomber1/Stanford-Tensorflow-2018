import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xlrd

DATA_FILE = 'data/fire_theft.xls'

book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# insert placeholders

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Creating the weight and bias variables

w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

# Creating a model

Y_pred = w * X + b

# Specifying the loss function

# loss = tf.square(Y_pred - Y, name='loss')

# The loss is heavily dependent on the last point..causing the graph to pull up
# Therefore we define Huber loss making it less dependent on those inflating points
# we cannot write if else loops in tensorflow, so we define a func
# Defining Huber loss


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = tf.square(residual) * 0.5
    large_res = (delta * residual) - (0.5 * tf.square(delta))
    return tf.where(condition, small_res, large_res)


# Specifying a new loss function
loss = huber_loss(Y, Y_pred, delta=1.0)

# Creating an optimizer

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# Training our model

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./graph/linear_reg", sess.graph)

    for i in range(100):
        total_loss = 0
        for x, y in data:
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += l
        print("Epoch {0}: {1}".format(i, total_loss / n_samples))

    writer.close()
    w_value, b_value = sess.run([w, b])

# Plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w_value + b_value, 'r', label='Predicted data')
plt.legend()
plt.show()

# We can see the loss decreasing by a large margin by using Huber loss
