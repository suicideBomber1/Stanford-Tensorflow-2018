import tensorflow as tf

x = tf.constant(10, name='x')
y = tf.constant(20, name='y')
z = tf.add(x, y)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(z)
writer.close()

# To visualize in tensorboard run the above python script and then run the below shell script in terminal
'''
tensorboard --logdir="./graphs" --port 6006 # or any other port
'''