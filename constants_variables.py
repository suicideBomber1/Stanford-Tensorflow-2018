### Tensorboard and constants ###

import tensorflow as tf

'''
# a = tf.constant(2, shape=[2, 2], verify_shape=True)
a = tf.constant(2, shape=[2, 2])
# b = tf.constant(3, name='b')

with tf.Session() as sess:
    print(sess.run(a))
'''
a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
x = tf.add(a, b)
y = tf.multiply(a, b)

with tf.Session() as sess:
    print(sess.run([x, y]))

# Similar to numpy

##### Creating a tensor with a particular value #####
# Constants as Sequences

a = tf.zeros([2, 3], dtype=tf.int32)
# Similarly tf.ones
# if no dtype is mentioned then tf.float32 is as default
# Again similar to numpy


# If we want the zeros tensor's shape to be of another tensor,
# then we define it as tf.zeros_like

input_tensor = tf.constant([[0, 1], [1, 2], [2, 3]])
z = tf.zeros_like(input_tensor)
# This will return the zeros tensor with the shape of input tensor
# Similar tf.ones_like

# If we want to fill it with values other than 0 & 1 we use tf.fill()

# tf.fill(dim, value, name=None)
# Value must be a scalar

d = tf.fill([2, 4], 8)

# Getting a sequence of numbers starting from a particular number

# We use tf.linspace() similar to np.linspace() except the input is float
# Here we indicate the number of elements between the start and stop
a = tf.linspace(30.0, 20.0, 11)

# tf.range(start, limit, delta)

a = tf.range(3, 18, 3)

# By now we understand that numpy and tensorflow work seamlessly
'''
***
Tensor objects are not iterable..
So, we cannot write
for _ in tf.range(3)
^ This would raise an error.
***
'''
test_tensor = tf.constant([[1, 2], [2, 3], [3, 4]])

x = tf.random_normal(shape=(), mean=0, stddev=1, dtype=tf.float32)
# This gives an random number from the normal distribution
# dtype cannot be tf.int32 (obviously)

y = tf.truncated_normal(shape=(), mean=0, stddev=1, dtype=tf.float32)
# Truncated normal is similar to random normal except it gives a number
# that is in between 2 standard deviations of mean

z = tf.random_uniform(shape=(), minval=2, maxval=5, dtype=tf.float32)
# Generates a number that is in the uniform distribution between minval and maxval

w = tf.random_shuffle(value=test_tensor)
# Randomly shuffles the given tensor only on the first dimension

x = tf.random_crop(value=test_tensor, size=(2, 2))
# Randomly crops the tensor in the given size

y = tf.multinomial(logits, n_samples)
#

z = tf.random_gamma(shape=[10], alpha=5)
# Draws shape samples from each of the given Gamma distribution(s).
# alpha is the shape parameter describing the shape of the distribution

w = tf.set_random_seed(seed)
# To set a random seed

# tf.add_n is for adding more than two elements
a = tf.constant([3, 6])
b = tf.constant([2, 2])
c = tf.add_n([a, b, b])  # For add_n: Input the elements as a list

# tf.reshape(tensor, shape) .... Used in case of multiplication

divisors = tf.div(a, b)  # Gives the divisors a // b
remainders = tf.mod(a, b)  # Gives the remainder a % b

# Tensorflow inderstands python native data types: Bool, int, float, str
t_0 = 19
x = tf.zeros_like(t_0)  # Gives single zero as output

t = ['apple', 'peach', 'banana']
x = tf.zeros_like(t)  # Meaning for zero in string is an empty str
y = tf.ones_like(t)
# This will throw an error as Tensorflow does not understand
# the meaning of one's in string ...(Makes sense !!): TypeError

# For booleans a zeros_like give all zeros and ones_like give all ones
'''
****************************************************************

////////////////////////// Variables  /////////////////////////////    '''

# Variable is a class and constant is an op.. Hence tf.Variable() and not tf.variable()

# Variable can be a scalar or vector or a matrix
a = tf.Variable(2, name='scalar')
b = tf.Variable([1, 2], name='vector')
c = tf.Variable(tf.zeros(784, 10))

# As mentioned before Variable is a class and has several inbuilt ops
# So to run these ops a session is to be called

# Easiest way to initialize all variables at once

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)


# Initializing a subset of variables

init_subset = tf.variables_intializer([a, b], name='init_subset')
with tf.Session() as sess:
    sess.run(init_subset)

# Single variable initializer...
# Every variable has an initializer op built in which can be used to initialize a single variable

w = tf.Variable(tf.zeros([784, 10]))
with tf.Session() as sess:
    sess.run(w.initializer)
    print(w.eval())

# Variable Assign

w = tf.Variable(10)
w.assign(100)

with tf.Session() as sess:
    sess.run(w.initializer)
    print(w.eval())

# This would not assign the value of 100 to w, since it is not called

# To make the assignment operation to work

w = tf.Variable(10)
w_assigned = w.assign(100)

with tf.Session() as sess:
    sess.run(w.initializer)
    sess.run(w_assigned)
    print(w.eval())

# Above we ran both the initializer and the assign op to perform the desired action
# We can skip the first sess.run(w.initializer) and just run the assign in session...
# Tensorflow understands w also need to be initialized.

w = tf.Variable(10)
w_assigned = w.assign(100)

with tf.Session() as sess:
    # sess.run(w.initializer)
    sess.run(w_assigned)
    print(w.eval())

# The above code also works
# Initializer op is the assign op that assigns the variable initial value to variable itself

my_var = tf.Variable(2, name='my_var')
my_var_times_two = my_var.assign(2 * my_var)

with tf.Session() as sess:
    sess.run(my_var_times_two)
    print(my_var_times_two)


my_var = tf.Variable(2, name='my_var')
my_var_times_two = my_var.assign(2 * my_var)

with tf.Session() as sess:
	sess.run(my_var)
    sess.run(my_var_times_two)
    sess.run(my_var_times_two)
    sess.run(my_var_times_two)
    sess.run(my_var_times_two)

# This results in my_var being multiplied by 2 everytime my_var_two is called



# assign_add() and assign_sub()
my_var = tf.Variable(10)

with tf.Session() as sess:
	sess.run(my_var.initializer)
	sess.run(my_var.assign_add(30))
	sess.run(my_var.assign_sub(40))

# The net result is zero

'''
***********
Unlike assign(), assign_add() & assign_sub() cannot initialize as they require the value of my_var
***********
'''

# Each session maintains its own copy of variable

my_var = tf.Variable(10)

sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(my_var.initializer)
sess1.run(my_var.assign_add(10))

sess2.run(my_var.initializer)
sess2.run(my_var.assign_sub(10))

sess1.close()
sess2.close()

# Initializing a variable with another variable

W = tf.Variable(tf.truncated_normal([700, 10]))
U = tf.Variable(2 * W)

# In the above case while running the session we need to initialize first "W" and then "U". Otherwise it throws an error
# To initialize variable "W" without calling initializer op we do the following

W = tf.Variable(tf.truncated_normal([700, 10]))
U = tf.Variable(2 * W.initialized_value())

############# Control Dependencies ###############

tf.Graph.control_dependencies(control_inputs)
'''
with g.control_dependencies([a, b, c]):
	d = ......
	e = ......
Here d & e will run only after a, b, c have been initialized
'''

################### PLACEHOLDERS ###################

# We want our clients to supply their own data later when they need to execute the program
# Placeholder will assemble the graph first without knowing the values needed for computation

'''
syntax = tf.placeholder(dtype, shape=None, name=None)

'''
# create a placeholder of type float 32-bit, shape is a vector of 3 elements
a = tf.placeholder(tf.float32, shape = [3])

# create a constant of type float 32-bit, shape is a vector of 3 elements
b = tf.constant([5, 6, 7], tf.float32)

# use the placeholder as you would a constant or a variable
c = a + b

# feed [1, 2, 3] to placeholder a via the dict {a: [1, 2, 3]}
# fetch value of c
with tf.Session as sess:
	sess.run(c, {a: [1, 2, 3]})


# If we shape as None then any shape is accepted as value of placeholder
# But debugging becomes difficult, since many ops require shape

# For feeding various values
with tf.Session as sess:
	for a_val in list_of_values_of_a:
		sess.run(c, {a: a_val})


# Normal loading vs Lazy loading

# Normal Loading
x = tf.constant(10)
y = tf.constant(20)
z = tf.add(x, y)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('./my_graph/12', sess.graph)
	for _ in range(10):
		sess.run(z)
	writer.close()


# Lazy laading
x = tf.constant(10)
y = tf.constant(20)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('./my_graph/12', sess.graph)
	for _ in range(10):
		sess.run(tf.add(x, y))
	writer.close()

# Although we saved one line of code we missed the node add in graph which makes it difficult
# Also, in normal loading the node add was created only once, while in lazy loading it was created multiple times in the loop.











































