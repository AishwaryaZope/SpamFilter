import pandas as pd       

x = pd.read_csv("train.csv")
x.head()

import nltk
nltk.download('punkt')

x['q_n_words'] = x['question_text'].apply(lambda row: len(row.split(" ")))

print(min(x["q_n_words"]))

print(max(x["q_n_words"]))

x[x['q_n_words']== 1]

x[x['q_n_words']== 122]

x = x.drop(labels=[522266], axis=0)

from string import punctuation 
punctuation
import re

x['question_text']=x['question_text'].apply(lambda x:re.sub("["+punctuation+"]",' ',x))

x['question_text']=x['question_text'].apply(lambda x:re.sub("\d",' ',x))

x['question_text']=x['question_text'].apply(lambda x:re.sub("\s+",' ',x))

x['question_text']=x['question_text'].str.lower()

x['question_text']

size = x.q_n_words.unique()

size = size.mean()

size
x.target.value_counts()

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Activation, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from numpy import array

from sklearn.model_selection import train_test_split


x['question_text']

text = x['question_text'].tolist()
text[:2]

y = x['target']

token = Tokenizer()
token.fit_on_texts(text)

vocab_size = len(token.word_index)+1
vocab_size

encoded_text = token.texts_to_sequences(text)

print(encoded_text[:30])

max_length = 33
X = pad_sequences(encoded_text, maxlen=max_length, padding="post")

print(X)

glove_vector = dict()

file = open('glove.6B.100d.txt', encoding= "utf-8")

for line in file:
  values = line.split()
  word = values[0]
  vectors = np.asarray(values[1:])
  glove_vector[word] = vectors
file.close()

len(glove_vector.keys())

glove_vector.get("else").shape

word_vector_matrix = np.zeros((vocab_size, 100))

for word, index in token.word_index.items():
  vectors = glove_vector.get(word)
  if vectors is not None:
    word_vector_matrix[index] = vectors
  else:
    print(word)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 19, test_size = 0.2, stratify = y)

from keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor = 'val_loss',
                          min_delta = 0,
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

vec_size = 100

model = Sequential()
model.add(Embedding(vocab_size, vec_size, input_length = max_length, weights = [word_vector_matrix], trainable = False))
model.add(Conv1D(64, 8, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.5))

model.add(Conv1D(32,8, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(16, activation='relu'))

model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer = Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'] )

model.summary()

model_history = model.fit(X_train, y_train, epochs = 30, validation_data = (X_test, y_test), callbacks = earlystop)

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
ax1.plot(model_history.history['loss'], color='b', label="Training loss")
ax1.plot(model_history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, 8, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(model_history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(model_history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, 8, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()

from tensorflow.keras.models import model_from_json

model_json = model.to_json()

with open("spam_filter.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights

model.save_weights("spam_filter.h5")
