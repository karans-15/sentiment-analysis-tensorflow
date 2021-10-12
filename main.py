# All required imports
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, LSTM, Bidirectional


df = pd.read_csv('amazon_baby.csv')
df['sentiments'] = df.rating.apply(lambda x: 0 if x in [1, 2] else 1)


# Splitting the dataset
split = round(len(df)*0.8)
train_reviews = df['review'][:split]
train_label = df['sentiments'][:split]
test_reviews = df['review'][split:]
test_label = df['sentiments'][split:]


# Convert all reviews to a string. Just a check if they arent in a string format already
import numpy as np
training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []
for row in train_reviews:
    training_sentences.append(str(row))
for row in train_label:
    training_labels.append(row)
for row in test_reviews:
    testing_sentences.append(str(row))
for row in test_label:
    testing_labels.append(row)


# Data processing parameters and hyper-parameters
vocab_size = 40000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = '<oov>'
padding_type = 'post'


# Tokenizing the words
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index


#Sentence to block of words
sequences = tokenizer.texts_to_sequences(training_sentences)  


# Padd if they are lesser than max length
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)


# Do the same for the test data
testing_sentences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sentences, maxlen=max_length)


model = Sequential()
model.add(Embedding(vocab_size, embedding_dim,input_length=max_length))
model.add(GlobalAveragePooling1D())
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


checkpoint_path = "model_checkpoint"
model.load_weights(checkpoint_path)


# Defining prediction function
def predict_sentiment(predict_msg):
    new_seq = tokenizer.texts_to_sequences(predict_msg)
    padded = pad_sequences(new_seq, maxlen =max_length,
                      padding = padding_type,
                      truncating=trunc_type)
    return (model.predict(padded))


def main():
    print("Welcom to Sentiment Analysis")
    print("===================================================================")
    while(True):
        user_input = input("Input string: ")
        prediction = np.array(predict_sentiment([user_input]))[0][0]
        if prediction>=0.5:
            print("Positive Feedback!")
        else:
            print("Negative feedback!")


        user_input = input('Do you want another prediction? (y/[n])? ')
        if user_input != 'y':
            print("bye!")
            break

if __name__ == '__main__':
    main()




