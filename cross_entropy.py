import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

cross_entropy = -tf.reduce_sum(tf.multiply(one_hot, tf.log(softmax)))
# Print cross entropy from session
with tf.Session() as sess:
    print(sess.run(cross_entropy, feed_dict = {one_hot: one_hot_data, softmax: softmax_data}))