# from keras.models import Sequential
# from keras.layers import Dense
# from keras.models import Model
# from keras.layers import Input
import pandas as pd
import statistics as st
from functools import partial

from ml_utils import *


class Par:
    
    def __init__(self):
        pass
    
    def pimba(self, a, b):
        a = a + b
        return a
    

par = Par()

p = partial(par.pimba, b=10)
a = p(2)

print(a)

# def monte_carlo(data, data_vol):
#     mean = st.mean(data)
#     stdev = st.stdev(data)
    
#     drift = mean - 0.5 * (stdev**2)
    
#     volatily = stdev * 
    
#     r = drift + volatily
    
    
# def multi_neural(inputs):
#     input = Input(shape=(360,45))
#     layer_1 = Dense(units=31, activation= 'relu')(input)
#     layer_2 = Dense(units=31, activation= 'relu')(layer_1)
#     layer_3 = Dense(units=17, activation='softmax')(layer_2)

# model = Model(input, layer_3)

# estimator.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# estimator.fit()

# Teste trends
# import pandas as pd                        
# from pytrends.request import TrendReq
# pytrend = TrendReq()

# kw_list=['Bitcoin', 'Cripto', 'Coin', 'BTC']

# pytrend.build_payload(kw_list=kw_list, timeframe= 'all')

# df = pytrend.interest_over_time()
# print(df)

# X_train = np.array([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12])
# y_train = np.array([4], [7], [10], [13])

# X_test = np.array([[7, 8, 9], [10, 11, 12]])
# y_test = [16]

# estimator = LassoLars()

# print(estimator.get_params())


# def soma(a = 0, b = 0):
#     c = a + b
#     return c
    
# params = {'a': 10, 'b' : 20,}

# r = soma(**params)
# print(r)

# model = estimator.fit(X_train, y_train)
# prediction = model.predict(X_test)

# print(prediction)


# bitcoin = Data_Model(interval="1d", period="2y")

# # scaler = bitcoin.processing_data()

# X_train, y_train, date = bitcoin.split_data()

# X_test, y_test, date = bitcoin.split_data(columns = ("Close", "Close", "Date"), offset = 2, total_len = 9, X_y_len = (7, 1))
# print(X_train.shape)
# estimator = ElasticNet()

# model = estimator.fit(X_train, y_train)
# prediction = model.predict(X_test)

# print(prediction, y_test, date)
# print(scaler.inverse_transform(prediction.reshape(-1, 1)), date)