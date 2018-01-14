## Graphs and Sessions

import tensorflow as tf

a = tf.add(3, 5)
print(a)

sess = tf.Session()
print sess.run(a)
sess.close()

# OR

with tf.Session as sess:
    print(sess.run(a))

# No need to close the session after opening as it is taken care of


x = 2
y = 3

op1 = tf.add(x, y)
op2 = tf.multiply(x, y)

# Adding a useless variable ... Not going to be calculated unless called upon in sessions
# Hence saves computation

useless = tf.multiply(x, add_op)

op3 = tf.pow(op2, op1)
with tf.Session() as sess:
    op3 = sess.run(op3)

# If we want multiple fetches in a session.. We need to pass the required items as a list in sess.run()
with tf.Session() as sess:
    z, not_useless = sess.run([op3, useless])

################## Sub graphs  #################

'''
Possible to break graphs into several chunks and run them parallelly
across multiple CPUs, GPUs, or devices

Ex: AlexNet

'''
# Distributed Computation is covered in Week 8.. So chill for now

# More than one graph (Not at all advised .. lets see)
'''

 1. Graphs are selfish and try to use up all the resources available by default
 2 .It's better to have disconnected subgraphs within one graph

 '''

# But if you still want to create separate graphs other than the default .. Heres how

# create a graph

g = tf.Graph()


# to add operators to the graph we need to set it as default
with g.as_default():
    x = tf.add(5, 11)

sess = tf.Session(graph=g)

sess.run()

sess.close()


# to handle default graphs

g = tf.get_default_graph()
# Now g is our default graph and whatever we add as default goes to g... and we need not mention g.as_default() repeatedly ...
# Although the downside is g is just a deafult one here...naming the default as g is itself loosing


# Do not mix default graphs and user created graphs

