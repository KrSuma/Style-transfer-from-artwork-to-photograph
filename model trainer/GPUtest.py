import tensorflow as tf

with tf.device('device:GPU:0'):
    a = tf.constant(1)
    b = tf.constant(2)
    c = a+b

# quick test to see if tensorflow-gpu recognizes the GPU
# this is important if you want a speedy process!
# if GPU is not used, then the CPU is used by default

sess = tf.Session()
sess.run(tf.global_variables_initializer())
