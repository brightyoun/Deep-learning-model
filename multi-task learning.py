import tensorflow as tf
import numpy as np

# Define placeholders
X = tf.placeholder("float", [10, 10], name="X")
Y1 = tf.placeholder("float", [10, 20], name="Y1")
Y2 = tf.placeholder("float", [10, 20], name="Y2")

# Define weights for layers
init_shared_layer_W = np.random.rand(10, 20)
init_Y1_layer_W = np.random.rand(20, 20)
init_Y2_layer_W = np.random.rand(20, 20)

shared_layer_W = tf.Variable(init_shared_layer_W, name="share_W", dtype="float32")
Y1_layer_W = tf.Variable(init_Y1_layer_W, name="Y1_W", dtype="float32")
Y2_layer_W = tf.Variable(init_Y2_layer_W, name="Y2_W", dtype="float32")

# Consturct layers with RELU activation
shared_layer = tf.nn.relu(tf.matmul(X, shared_layer_W))
Y1_layer = tf.nn.relu(tf.matmul(shared_layer, Y1_layer_W))
Y2_layer = tf.nn.relu(tf.matmul(shared_layer, Y2_layer_W))

# Calculate loss
Y1_loss = tf.nn.l2_loss(Y1 - Y1_layer)
Y2_loss = tf.nn.l2_loss(Y2 - Y2_layer)
Joint_loss = Y1_loss + Y2_loss

# Optimizers
Y1_op = tf.train.AdamOptimizer().minimize(Y1_loss)
Y2_op = tf.train.AdamOptimizer().minimize(Y2_loss)
Optimizer = tf.train.AdamOptimizer().minimize(Joint_loss)

#====================================================================================================
# Joint Training
# Session code
#====================================================================================================

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for iters in range(10) :
        _, Joint_loss_res = sess.run([Optimizer, Joint_loss], {
            X: np.random.rand(10, 10)*10,
            Y1: np.random.rand(10, 20)*10,
            Y2: np.random.rand(10, 20)*10 })
        print(Joint_loss_res)


# Joint Training
