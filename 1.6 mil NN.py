import tensorflow as tf
import numpy as np
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
n_nodes_hl1 = 500
n_nodes_hl2 = 500
#n_nodes_hl3 = 500
lemmatizer = WordNetLemmatizer()
n_classes = 2
batch_size = 32
total_batches = int(1600000/batch_size)
hm_epochs = 10
hm_data = 2000000
x = tf.placeholder('float', (2638, None))
y = tf.placeholder('float')

hidden_1_layer = {'f_fum': n_nodes_hl1,
                      'weight': tf.Variable(tf.random_normal([2638, n_nodes_hl1])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum': n_nodes_hl2,
                    'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                    'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum': None,
                'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias': tf.Variable(tf.random_normal([n_classes])),}
def neural_network(data):
    layer_1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    layer_2 = tf.nn.relu(layer_2)
    # temp is t
    output = tf.matmul(layer_2, output_layer['weight']) + output_layer['bias']

    return output

saver = tf.train.Saver()
tf_log = 'tf.log'
#THIS IS HOW WE WOULD BE SAVING OUR MODEL IN FORM OF CHECKPOINTS

# training our neural network
def train_neural_network(x):
    prediction = neural_network(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    #n_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # initializes our variables. Session has now begun.
        try:
            epoch = int(open(tf_log,'r').read().split('\n')[-2])+1
            print('Starting:',epoch)  ##This is our wat of tracking hich epoch we
                    # are on using a log ifle
            #epoch_loss = 0  # we'll calculate the loss as we go
        except:
            epoch = 1
            #this is our way of continuing until we have got as many epochs as pssobile
        while epoch<=hm_epochs:
            if epoch!=1:  # if we are not starting at the first one, we load the checkpoint file created earlier
                saver.restore(sess, "model.ckpt")
            epoch_loss = 1
            # we load in the lexicon pickle
            with open('lexicon.pickle','rb') as f:
                lexicon = pickle.load(f)
            #we load in the shuffled training set


            with open('train_set_shuffled.csv', buffering=20000, encoding = 'latin-1') as f:
                batch_x = []
                batch_y = []
                batches_run = 0
        # from here, each line will be vectorized and added to our batch
                #which will be size of the buffers we are creating
                for line in f:
                    label = line.split(':::')[0]
                    tweet = line.split(':::')[1]
                    current_words = word_tokenize(tweet.lower())
                    current_words = [lemmatizer.lemmatize(i) for i in current_words]

                    features = np.zeros(len(lexicon))

                    for word in current_words:
                        if word.lower() in lexicon:
                            index_value = lexicon.index(word.lower())

                            features[index_value]+=1
                    line_x = list(features)
                    line_y = eval(label)

                    batch_x.append(line_x)
                    batch_y.append(line_y)

                    if len(batch_x)>=batch_size:
                        _, c=sess.run([optimizer, cost], feed_dict={x: np.array(batch_x),
                                                                    y: np.array(batch_y)})
                        epoch_loss +=c
                        batch_x = []
                        batch_y = []
                        batches_run+=1
                        print('Batch run:', batches_run, '/', total_batches,'| Epoch:', epoch,'| Batch Loss:',c,)
            #at this point, it's a full EPOCH so we go ahead and save it
            # at the same time, we update our EPOCH log file
            saver.save(sess, "model.ckpt")
            print('Epoch',epoch,'completed out of',hm_epochs, 'loss:', epoch_loss)

            with open(tf_log, 'a') as f:
                f.write(str(epoch)+ '\n')
            epoch+=1
train_neural_network(x)

## Since we are using checkpoints and a larger dataset,
        # we might need a separate test function for accuracy

def test_neural_network():
    prediction = neural_network(x)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            try:
                saver.restore(sess, "model.ckpt")

            except Exception as e:
                print(str(e))

            epoch_loss = 0

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        feature_sets = []
        labels = []
        counter = 0

        with open('processed-test-set.csv', buffering=20000) as f:
            for line in f:
                try:
                    features = list(eval(line.split('::')[0]))
                    label = list(eval(line.split('::')[1]))
                    #print(features)
                    #print(label)

                    feature_sets.append(features)
                    labels.append(label)
                    counter+=1

                except:
                    pass
        print('Tested', counter, 'samples.')


        test_x = np.array(feature_sets)
        test_y = np.array(labels)

        print('Accuracy:', accuracy.eval({x:test_x, y:test_y}))

test_neural_network()


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

'''
Accuracy of approximately 96% 
Epoch 0 completed out of 10 loss: 65.5630090833
Epoch 1 completed out of 10 loss: 48.9673840702
Epoch 2 completed out of 10 loss: 37.7931989431
Epoch 3 completed out of 10 loss: 25.6732599437
Epoch 4 completed out of 10 loss: 21.6604651734
Epoch 5 completed out of 10 loss: 19.5112599507
Epoch 6 completed out of 10 loss: 15.0463630259
Epoch 7 completed out of 10 loss: 22.6016843393
Epoch 8 completed out of 10 loss: 18.3015118763
Epoch 9 completed out of 10 loss: 10.1417467222
Accuracy: 0.96113


'''

#### MUCH LARGER DATA SET IN NEXT TUTORIAL #####

























