import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json

base = pd.read_csv('iris.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

classificador = Sequential()
classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform', input_dim = 4))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 3, activation = 'softmax'))
classificador.compile(optimizer = 'adam', 
                          loss = 'sparse_categorical_crossentropy', 
                          metrics = ['accuracy'])
classificador.fit(previsores,classe,
                  batch_size=10, epochs=2000)

classificador_json = classificador.to_json()
with open('classificador_iris.json', 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('classificador_iris.h5')

novo = np.array([[3.2, 4.5, 0.9, 1.1]])
previsao = classificador.predict(novo)
previsao = (previsao > 0.5)

if previsao[0][0] == True and previsao[0][1] == False and previsao[0][2] == False:
    print('Iris setosa')
elif previsao[0][0] == False and previsao[0][1] == True and previsao[0][2] == False:
    print('Iris virginica')
elif previsao[0][0] == False and previsao[0][1] == False and previsao[0][2] == True:
    print('Iris versicolor')
    
classificador.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
resultado = classificador.evaluate(previsores,classe_dummy)

