# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 18:53:56 2019

@author: Souparna
"""
import gensim
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.model_selection import train_test_split


from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Flatten
from keras.layers import Embedding, Bidirectional
from keras.layers.recurrent import LSTM


from utilities import load_embedding, get_weight_matrix

import pandas as pd
data_file_path = "G:/MER_data/CADEC/CADEC.csv"

w2v_filename = "../w2v_embeddings.txt"

data = pd.read_csv(data_file_path)
print(data["Tag"].value_counts())


#------------------ word 2 vec ----------------------
sentences = data.groupby(["Sentence No"])["Word"].apply(list).to_list()

texts = [gensim.utils.simple_preprocess(i) for word in sentences for i in word ]

#w2v_model = Word2Vec(texts, min_count = 1,  size = 300, window = 5, iter = 50)
#w2v_model.wv.save_word2vec_format(w2v_filename, binary=False)
#------------------------------------------------------
#text = []
#sent = ""
#for w in data["Word"]:
#    if w == "0":
#        text.append(sent + "0")
#        sent = ""
#    else:
#        sent += w + " "

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

max_length = max([len(i) for i in sentences])

encoded_docs = tokenizer.texts_to_sequences(sentences)
X_data = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

vocab_size = len(tokenizer.word_index) + 1


# load embedding from file
raw_embedding = load_embedding(w2v_filename)
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index, 300)
# create the embedding layer
embedding_layer = Embedding(vocab_size, 300, weights=[embedding_vectors],
                            input_length=max_length, trainable=False)


y_labels = data.groupby(["Sentence No"])["Tag"].apply(list).to_list()



mlb = MultiLabelBinarizer()
y_labels = mlb.fit_transform(y_labels)

x_train, x_test, y_train, y_test = train_test_split(X_data, y_labels, test_size = 0.2)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(300)))
model.add(Dense(11))   
model.add(Activation('softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy')
print(model.summary())

history = model.fit(x_train, y_train, batch_size=2048, epochs=20,validation_split=0.1, verbose=2)


