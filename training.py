import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetlemmatizer #Para pasar las palabras a su forma raiz

#Para crear la red neuronal
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import sgd_experimental

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json").read())  
                          
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4")
words=[]
classes=[]
documents=[]
ignore letters = ['?','!',',','.']
                  
#Clasifica los patrones y las categorias
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_ list= nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append[(word_list; intent!"tag")
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatizelword) for word in words if word not in ignore_letters]
words= sorted(set(words))

pickle.dump(words, open('words.pkl", 'wb'))
pickle.dump(classes, open('classes.pkl",'wb"))
                          
#Pasa la información a unos y ceros según las palabras presentes en cada categoría para hacer el Y entrenamiento
training=[]
output_empty = [0]*len(classes)
for document in documents:
    bag=[]
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_ patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output row = listfoutput_ empty)
    output _ row[classes.index(document[1])] =1
    training.append([bag, output_row])
random.shuffle(training)

training= np.array(training)
print(training)

#Reparte los datos para pasarlos a la red
train_x = list(training[:,0])
train _y =list(training[:,1])

#Creamos agentes inteligente con apoyo de una red neuronal
model = Sequential()
nodel.add(Dense(128, input _shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train _ [0]), activation='softmax")
#Creamos el optimizador y lo compilamos
sgd = sgd _ experimental.SGD(learning_ rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model. compile(loss='categorical_ crossentropy', optimizer = sgd, metrics = ['accuracy'])
#Entrenamos el modelo y lo guardamos
train _ process = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch size=5,
verbose=1)
model.save("chatbot _model.h5", train_process)