import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

base = pd.read_csv('iris.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criar_rede():
    classificador = Sequential()
    classificador.add(Dense(units = 16, activation = 'relu', 
                            kernel_initializer = 'random_uniforme', input_dim = 4))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 16, activation = 'relu', 
                            kernel_initializer = 'random_uniforme'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 3, activation = 'softmax'))
    classificador.compile(optimizer = optimizer, 
                          loss = 'sparse_categorical_crossentropy', 
                          metrics = ['accuracy'])
    return classificador

classificador = KerasClassifier(build_fn=criar_rede,
                                epochs=2000,
                                batch_size=10)

resultados = cross_val_score(estimator = classificador,
                             X= previsores,y=classe,
                             cv=10,scoring='accuracy')

media = resultados.mean()
desvio = resultados.std()