import tensorflow as tf

# data: set of 60,000 records of training examples of handwritten digits
'''
    input > weight > hidden layer 1 (activation function) > weights > hidden layer 2
    HL2 > activation function > weights > output layer

     feed forward NN - data passed straight through 
                        - at end, o/p compared to intended o.p
                            using cost/loss function(ex: cross entropy)
                     - Optimization function(optimizer) used
                            to minimize the cost(AdamOptimizer, SGD, AdaGrad)
      Backpropagation - Opt. function goes backwards and manipulates the weights

    feed forward +_backprop = epoch(1 cycle) 
                            hope: with each cycle, we are lowering the cost function

    '''
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# one hot parameter - one component will be hot(dictates)
# rest are off # can be usefull for multi-class classification
# 10 classes, 0-9
# ONE HOT SITUATION
# 0 = [1,0,0,0,0,0,0,0,0,0]
# 1 = [0,1,0,0,0,0,0,0,0,0]
# 2 = [0,0,1,0,0,0,0,0,0,0]
# ... so on
# ONE HOT SITUATION
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100  # can do batches of 100 images at a time

# define some placeholding variables
# height x width
x = tf.placeholder('float', [None, 784])  # 28X28 = 784 pizels
# if we attempt to feed something not of None, 784 shape
# into x, TF will throw an error
y = tf.placeholder('float')


def neural_network(data):
    hidden_1_layer = {'weights': tf.Variable(tf.truncated_normal([784, n_nodes_hl1], stddev=0.1)),
                      'biases': tf.Variable(tf.constant(0.1, shape=[n_nodes_hl1]))}
    # this will create an array of weights in one giant tensor
    # biases are something that's added in after the weights
    # input data is multiplied by weights then added
    # biases role:   (input_data*weights) + biases
    # benefit of bias : if all input data is 0, i*w = 0 so we have bias
    # to make sure that some neurons could still fire
    hidden_2_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2], stddev=0.1)),
                      'biases': tf.Variable(tf.constant(0.1, shape=[n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3], stddev=0.1)),
                      'biases': tf.Variable(tf.constant(0.1, shape=[n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl3, n_classes], stddev=0.1)),
                    'biases': tf.Variable(tf.constant(0.1, shape=[n_classes])), }
    # no of biases for output layer = number of classes


    # (input_data * weights) + biases  IS THE MODEL WE HAVE for each layer
    layer_1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    # now goes through an activation function - sigmoid function
    layer_1 = tf.nn.relu(layer_1)
    # input for layer 2 = result of activ_func for layer 1
    layer_2 = tf.add(tf.matmul(layer_1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    layer_3 = tf.nn.relu(layer_3)

    output = tf.matmul(layer_3, output_layer['weights']) + output_layer['biases']

    return output

    # now all we have to do is explain to TF, what to do with this model
    # need to specify how we want to run data through that model


def train_neural_network(x):
    prediction = neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # using cross entropy as logits as our cost
    # calculate the difference of prediction that we got to the known label that we have
    # it's fine even if we aren't using one hot
    # output will always be in shape of training and testing set labels.
    '''AIM : MINIMIZE THE COST '''
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # adam optimizer = 1 parameter = learning_rate = 0.001(default) - won't be modifying

    n_epochs = 10  # cycles of feed forward + backprop

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # initializes our variables. Session has now begun.

        for epoch in range(n_epochs):
            epoch_loss = 0  # we'll calculate the loss as we go
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)

#### ACCURACY #####
'''
Extracting /
tmp/data/train-images-idx3-ubyte.gz
Extracting /tmp/data/train-labels-idx1-ubyte.gz
Extracting /tmp/data/t10k-images-idx3-ubyte.gz
Extracting /tmp/data/t10k-labels-idx1-ubyte.gz
Epoch 0 completed out of 10 loss: 1504538.08374
Epoch 1 completed out of 10 loss: 372819.888851
Epoch 2 completed out of 10 loss: 200020.001994
Epoch 3 completed out of 10 loss: 118924.167567
Epoch 4 completed out of 10 loss: 71611.6929203
Epoch 5 completed out of 10 loss: 45945.6708183
Epoch 6 completed out of 10 loss: 31109.7455745
Epoch 7 completed out of 10 loss: 21754.9216321
Epoch 8 completed out of 10 loss: 20885.2626089
Epoch 9 completed out of 10 loss: 18218.7463598
Accuracy: 0.9484
'''























