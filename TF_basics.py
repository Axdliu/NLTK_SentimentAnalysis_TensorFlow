'''
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
'''
# code kind of set up in the background. Everything runs in a chunk.
# python is kind of slow.
# Tflow takes stuff; goes into the background. Then comes to the front with the results.

# tensorflow is a library.

# these are basically array manipulation libraries.
# TF is actually a deep learning library.
import tensorflow as tf
# note: tf.mul = tf.multiply
# tf.sub - tf.subtract()
# tf.neg - tf.negative()
x1 = tf.constant(5)
x2 = tf.constant(6)
result = tf.multiply(x1, x2)
# is same as result = x1*x2
print(result)
    # result is an abstract tensor in our computation graph
    # to see the result run it in a session
with tf.Session() as sess:
    output = sess.run(result)
    print(output) # automatically close sun
#print(output)
    ## OR ##
    #COMPUTATION GRAPH - ABSTRACT GRAPH
    # session - can output stuff/modify tensors within that graph
#print(sess.run(result))# will throw error since outside session
'''sess = tf.Session()
print(sess.run(result))
sess.close()'''

