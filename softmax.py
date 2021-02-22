# Solution is available in the other "solution.py" tab
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)
    
    # Calculate the softmax of the logits as an input and give the softmax activations back
    softmax = tf.nn.softmax(logits)
    
    with tf.Session() as sess:
        # Feed in the logit data
        output = sess.run(softmax, feed_dict={logits : logit_data})

    return output
