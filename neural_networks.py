from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import LinearRegression
from storage import *

# Teste trends
import pandas as pd                        
from pytrends.request import TrendReq
pytrend = TrendReq()

kw_list=['Bitcoin', 'Cripto', 'Coin', 'BTC']

pytrend.build_payload(kw_list=kw_list, timeframe= 'all')

df = pytrend.interest_over_time()
print(df)

X_train = np.array([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12])
y_train = np.array([4], [7], [10], [13])

# X_test = np.array([[7, 8, 9], [10, 11, 12]])
# y_test = [16]

# estimator = LinearRegression()

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













