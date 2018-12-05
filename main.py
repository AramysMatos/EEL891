import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np


def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

def get_index_to_remove_outliers(x):
    outlierConstant = 1.5

    col = np.array(x)
    upper_quartile = np.percentile(col, 75)
    lower_quartile = np.percentile(col, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for index, value in enumerate(col.tolist()):
        if value <= quartileSet[0] or value >= quartileSet[1]:
            resultList.append(index)
    return resultList

def cleaning_data(dataset, dataset_test):

    len_train = dataset.shape[0]
    len_test = dataset_test.shape[0]

    frames = [dataset, dataset_test]
    result = pd.concat(frames)

    # removendo as colunas: id, descrição, bairro, preço
    X = result.iloc[:, [1,2,4,6,7,8,10,11,12,13,14,15,16,17,18,19,20]].values
    Y = result.iloc[:, 9].values

    # primeiramente codificar em valores inteiros dos campos tipo e tipo_vendedor
    labelencoder = LabelEncoder()

    for index in [13, 14]:
        X[:, index] = labelencoder.fit_transform(X[:, index])

    # em seguida criar uma variavel binaria para cada fit_transform
    onehotencoder = OneHotEncoder(categorical_features=[13, 14])
    X = onehotencoder.fit_transform(X).toarray()

    X_train = X[:len_train]
    X_test = X[len_train:]

    # pegando a coluna de preço
    y_train = Y[:len_train]

    # Removendo outlier de preço
    outliers = get_index_to_remove_outliers(y_train)
    X_train = np.delete(X_train, outliers, 0)
    y_train = np.delete(y_train, outliers, 0)


    return X_train, y_train, X_test

#Ler o dataset
dataset = pd.read_csv('data/train.csv')
dataset_test = pd.read_csv('data/test.csv')
dataset_sample = pd.read_csv('data/sampleSubmission.csv')

X_train, y_train, X_test = cleaning_data(dataset, dataset_test)

y_test = dataset_sample.iloc[:,-1]

#
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#------------------------------------------------------------------------------
# Treinar o regressor linear usando o conjunto de treinamento
#------------------------------------------------------------------------------

regressor.fit(X_train, y_train)

#------------------------------------------------------------------------------
#  Obter respostas do modelo para os conjuntos de treinamento e de teste
#------------------------------------------------------------------------------

y_pred_train = regressor.predict(X_train)
y_pred_test  = regressor.predict(X_test)

#------------------------------------------------------------------------------
#  Verificar desempenho do regressor
#     - nos conjunto de treinamento ("in-sample")
#     - nos conjunto de teste   ("out-of-sample")
#------------------------------------------------------------------------------

import math
from sklearn.metrics import mean_squared_error, r2_score

print('\nDesempenho no conjunto de treinamento:')
#print('MSE  = %.3f' %           mean_squared_error(y_train, y_pred_train) )
print('RMSE = %.3f' % rmspe(y_train, y_pred_train))
print('R2   = %.3f' %                     r2_score(y_train, y_pred_train) )

print('\nDesempenho no conjunto de teste:')
#print('MSE  = %.3f' %           mean_squared_error(y_test , y_pred_test) )
print('RMSE = %.3f' % rmspe(y_test , y_pred_test))
print('R2   = %.3f' %                     r2_score(y_test , y_pred_test) )

#------------------------------------------------------------------------------
#  Verificar os parâmetros do regressor
#------------------------------------------------------------------------------

import numpy as np

print('\nParametros do regressor:\n',
      np.append( regressor.intercept_ , regressor.coef_  ) )









