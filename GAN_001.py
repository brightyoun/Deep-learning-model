import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

#============================
# Option setting
#============================
total_epoch = 100
batch_size = 16
learning_rate = 0.0001

#============================
# Network layer option
#============================
n_hidden = 256
n_input = 28 * 28
n_noise = 200

#============================
# Network model
#============================
X = tf.placeholder(tf.float32, [None, n_input], name="X")
Z = tf.placeholder(tf.float32, [None, n_noise], name="Z")

# Generator model
# Init
init_G_W1 = tf.random_normal([n_noise, n_hidden], stddev=0.01)
init_G_b1 = tf.zeros([n_hidden])
init_G_W2 = tf.random_normal([n_hidden, n_input], stddev=0.01)
init_G_b2 = tf.zeros([n_input])

# component
G_W1 = tf.Variable(init_G_W1, name="G_W1", dtype="float32")
G_b1 = tf.Variable(init_G_b1, name="G_b1", dtype="float32")
G_W2 = tf.Variable(init_G_W2, name="G_W2", dtype="float32")
G_b2 = tf.Variable(init_G_b2, name="G_b2", dtype="float32")

# Discriminator model
# Init
init_D_W1 = tf.random_normal([n_input, n_hidden], stddev=0.01)
init_D_b1 = tf.zeros([n_hidden])
init_D_W2 = tf.random_normal([n_hidden, 1], stddev=0.01)
init_D_b2 = tf.zeros([1])

# componet
D_W1 = tf.Variable(init_D_W1, name="D_W1", dtype="float32")
D_b1 = tf.Variable(init_D_b1, name="D_b1", dtype="float32")
D_W2 = tf.Variable(init_D_W2, name="D_W2", dtype="float32")
D_b2 = tf.Variable(init_D_b2, name="D_b2", dtype="float32")

#===========================
# Network link
#===========================

# Generator
def generator(noize_z):
    hidden = tf.nn.relu(tf.matmul(noize_z, G_W1) + G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + G_b2)

    return output

# Discriminator
def discriminator(inputs):
    hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2)

    return output

def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

# Run Environments
G = generator(Z)
D_gene = discriminator(G)
D_real = discriminator(X)

# Calculate loss
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
loss_G = tf.reduce_mean(tf.log(D_gene))

# Var list
D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

# Optimizers
train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)

#==========================================================================================
# Model Train
#==========================================================================================

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples/batch_size)

    # loss variable
    loss_val_D = 0
    loss_val_G = 0

    for epoch in range(total_epoch):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            noise = get_noise(batch_size, n_noise)

            # Training
            _, loss_val_D = sess.run([train_D, loss_D],
                                     feed_dict={X:batch_xs, Z:noise})
            _, loss_val_G = sess.run([train_G, loss_G],
                                     feed_dict={Z:noise})

        print('Epoch : ', '%04d' % epoch,
              'D loss : {:.4}'.format(loss_val_D),
              'G loss : {:.4}'.format(loss_val_G))

        #======================================================================
        # Save res image
        #======================================================================
        if epoch == 0 or (epoch + 1) % 10 == 0:
            # generate 10 images
            sample_size = 10

            # After train model, execute generator model
            noise = get_noise(sample_size, n_noise)
            # run trained generator model
            samples = sess.run(G, feed_dict={Z:noise})

            # visualize
            fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

            for i in range(sample_size):
                ax[i].set_axis_off()
                ax[i].imshow(np.reshape(samples[i], (28, 28)))

            plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

print('End of train')
