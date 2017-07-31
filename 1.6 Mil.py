# sentiment analysis on a data set of 1.6 million samples ##
## sentiment 140 data set from Stanford ##

# at the moment, this data set might not be too large
    # but once it is converted into the bag of words model from before
'''
few changes when working with large data:
1. we want to buffer the incoming data into an acceptable size. 
2. Instead of reading the entire file at once, we can read the file
    in segments of 10 mega bytes at a time with buffering
3. we need to run the data through our network as well in batches. In the example 
 we discussed earlier, this wasn't the criteria. 
4. Now since training is taking much longer, we must be able to save our progress for
both purposes of continuity where we left off and also so that we need not re-train 
our model every time we wish to perform the analysis. 
'''

# we need to re-pre-process our data henceforth
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd


lemmatizer = WordNetLemmatizer()
''' 
polarity 0 = -ve 
polarity 2 = neutral 
polarity 4 = positive , ID, DATE, QUERY, USER, TWEET 
'''
# firstly we need to convert the sentiment values of the data set


def init_process(fin, fout):
    outfile = open(fout,'a')
    with open(fin, buffering=200000, encoding='latin-1') as f:
        try:
            for line in f:
                line = line.replace('"','')
                init_polarity = line.split(',')[0]
                if init_polarity=='0':
                    init_polarity == [1,0] #-ve 1 +ve 0
                elif init_polarity=='4':
                    init_polarity = [0,1]

                tweet = line.split(',')[-1]
                outline = str(init_polarity)+':::'+tweet
                outfile.write(outline)
        except Exception as e:
            print(str(e))
    outfile.close()

# we pass a file then output the new file.
#we're modifying the sentiment label #only need to run this function once for training and testing data

init_process('training.1600000.processed.noemoticon.csv', 'train_set.csv')
init_process('testdata.manual.2009.06.14.csv', 'test_set.csv')

## now we need to create our lexicon

def create_lexicon(fin):
    lexicon = []
    with open(fin, 'r', buffering=100000, encoding ='latin-1') as f:
        try:
            counter = 1
            content = ''
            for line in f:
                counter+=1
                if(counter/2500.0).is_integer():
                    tweet=line.split(':::')[1]

                    content+= ' '+tweet
                    words = word_tokenize(content)
                    words = [lemmatizer.lemmatize(i) for i in words]
                    lexicon = list(set(lexicon + words))
                    print(counter, len(lexicon))
        except Exception as e:
            print(str(e))
    with open('lexicon.pickle', 'wb') as f:
        pickle.dump(lexicon, f)

create_lexicon('train_set.csv')

# now we need to vectorize the data into the bag of words model before we
    # put it into training the network or we can do it inline within the network
# could be possible for us to vectorize the data first, saving it somewhere and then
        #feeding it through the network
def convert_to_vec(fin, fout, lexicon_pickle):
    with open(lexicon_pickle, 'rb') as f:
        lexicon = pickle.load(f)
    outfile = open(fout, 'a')
    with open(fin, buffering= 20000, encoding = 'latin-1') as f:
        counter = 0
        for line in f:
            counter +=1
            label = line.split(':::')[0]
            tweet = line.split(':::')[1]
            current_words = word_tokenize(tweet.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]

            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] +=1
            features = list(features)
            outline = str(features)+'::'+str(label)+ '\n'
            outfile.write(outline)
        print(counter)

convert_to_vec('test_set.csv', 'processed-test-set.csv', 'lexicon.pickle')

def shuffle_data(fin):
    df = pd.read_csv(fin, error_bad_lines=False)
    df = df.iloc[np.random.permutation(len(df))]
    print(df.head())
    df.to_csv('train_set_shuffled.csv', index= False)
shuffle_data('train_set.csv')

def create_test_data_pickle(fin):
    feature_sets = []
    labels = []
    counter = 0
    with open(fin, buffering=20000) as f:
        for line in f:
            try:
                features = list(eval(line.split('::')[0]))
                label = list(eval(line.split('::')[1]))

                feature_sets.append(features)
                labels.append(label)
                counter+=1
            except:
                pass
    print(counter)
    feature_sets = np.array(feature_sets)
    labels = np.array(labels)
create_test_data_pickle('processed-test-set.csv')



#### NOW WE HEAD OVER TO OUR NN SCRIPT######



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




































