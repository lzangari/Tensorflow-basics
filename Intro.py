import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
#tf.compat.v1.disable_eager_execution()
# creating an object which is called tensor --> data rncapsulated in an object  but cannot store as integers, floats or strings

tensor_object = tf.constant('Hello World')   # zero dimentional string tensor   #in constant tensor the value cannot be changed
a = tf.constant(1234) # zero dimensional int32 tensor
b = tf.constant([123, 456, 789]) #one dimensional int32 tensor
c = tf.constant([[123, 456, 789], [222, 444, 555]]) #two dimensional int32 tensor
x = tf.placeholder(tf.string) #if we have non-constant tensor
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.compat.v1.Session() as sess: # the session is allocating the operation either in cpu or gpu

    # run the tf constant operation in the session
    output1 = sess.run(tensor_object)
    output2 = sess.run(x, feed_dict={x: 'Test', y: 123, z: 45.3})
    print(output2)


