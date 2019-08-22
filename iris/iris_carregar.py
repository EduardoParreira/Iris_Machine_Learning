import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

arquivo = open('classificador_iris.json','r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_iris.h5')

novo = np.array([[6.1, 4.0, 2.0, 0.5]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.5)

base = pd.read_csv('iris.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

classificador.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
resultado = classificador.evaluate(previsores,classe_dummy)