import spacy
import keras
from keras.utils import pad_sequences
import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Embedding,Dense, Dropout,SimpleRNN
from keras.losses import sparse_categorical_crossentropy


data=pd.read_csv(r"C:\Users\harii\Downloads\BOOK.csv")

data=data.sample(frac=1)

nlp = spacy.load("en_core_web_sm")

def preprocess(complaints):
    complaints = [nlp(i) for i in complaints]
    W=[]
    for sentence in complaints:
        temp=[]
        words = [i.lemma_ for i in sentence]
        words=[i for i in words if i not in [",",".","!","-"]]
        for word in words:
            if(nlp.vocab[str(word)].is_stop==True):
                temp.append(word)
        words=[i for i in words if i not in temp] 
        W.append(words)
    return W
W=preprocess(data["Complaint"])

c=1 #0 -> OOV
d={}
X=[]
for i in W:
    temp=[]
    for j in i:
        if j not in d.keys():
            d[j]=c
            c+=1
        temp.append(d[j])
    X.append(temp)
    
def label(query,mapping=d):
    encoded=[]
    for i in query:
        temp=[]
        for j in i:
            print(j)
            if (j in mapping.keys()):
                temp.append(mapping[j])
            else:
                temp.append(0)
        encoded.append(temp)
    return encoded

# import numpy as np
# X=np.array(X)

def pad(max_words,X):
    return pad_sequences(X,maxlen=max_words)
x_train=pad(30,X)
embedding_size=16

y = data['Domain'].astype("category")
y=y.cat.codes
y=np.array(y)

y_text = np.array(data['Domain'])

out_val_mapp = {}
for i in range(len(y)):
    if y[i] not in out_val_mapp.keys():
        out_val_mapp[y[i]] = y_text[i]
        
vocabulary_size = 600
model1=Sequential()
model1.add(Embedding(vocabulary_size, embedding_size, input_length=30))
model1.add(SimpleRNN(50))
model1.add(Dense(3, activation='sigmoid'))

model1.compile(loss=sparse_categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

model1.fit(x_train,y,epochs=10)

#%%

import pickle

pickle.dump(model1,open('model.pkl','wb'))








