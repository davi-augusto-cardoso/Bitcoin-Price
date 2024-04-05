# Version 1.0

from storage import Data_Model
from nostradamus import Nostradamus
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Dropout
from view import View

# TEMP IMPORTS
from estimators_utils import *
from spaces_utils import *
# from prophet import Prophet
# import statistics

# Organizar código - OK
# Normalizar dados e implementar algoritimos com a normalização dos dados
# Criar visualização e forma de comparação entre dados +- OK
# Criar conexão com API do google Trends - Não conseguir conectar
# Implementar redes neurais correlacioando os dados do google trends com histórico de preços
# Implementar monte carlo para predição de cenários futuros
# Implementar tunning de parametros
# Criar classificador que esolhe melhor modelo para calcular o dia seguinte

bitcoin = Data_Model(interval="1d", period="2y")

bitcoin.processing_data('Close', 'Volatil')
# scaler = bitcoin.filter_data('Close')
# scaler = bitcoin.filter_data('Mean HLC')
# scaler1 = bitcoin.filter_data('Low')
# scaler = bitcoin.filter_data('Close')

ndms = Nostradamus(bitcoin)

train_test_data = bitcoin.split_train_test(amount_tests = 30, train_len = 360, X_y_len = (45, 1)
                                        , columns = ("Close", "Close", "Date"))

# estimators_and_results = ndms.train_model(best_estimators, train_test_data)
estimators_and_results = ndms.train_tunning_model(estimators_hp, train_test_data)
performance_i = ndms.performance_individual(estimators_and_results)
# performance_g = ndms.performance_general(performance_i, estimators_and_results)
print(estimators_and_results)
view = View()
# view.ShowResults(estimators_and_results)
view.show_graphics(estimators_and_results)

# print(result)

# train_test_data = bitcoin.cutting_slices("Close", "Close", len_data=45, times=20, offset_X=8,offset_y=7)

# estimators_and_results = ndms.train_model(best_estimators, train_test_data)

# view = View()
# view.show_graphics(estimators_and_results, scaler)

# performance = ndms.performance_general(estimators_and_results)

# best_estimators = ndms.take_best_estimators(performance)

# ensamble = ndms.ensamble_estimators(best_estimators)

# estimators_and_results = ndms.train_model(ensamble, train_test_data)

# performance_ensamble = ndms.performance_general(estimators_and_results)
# print(estimators_and_results)
# view.show_graphics(estimators_and_results, scaler)

# profit = ndms.take_decision(estimators_and_results, train_test_data)

# print(profit)
# # print(models)

# perf_general = ndms.performance_general(models)

# # print(perf_general)

# mean_perf = ndms.mean_performance_estimators(perf_general)

# # print(mean_perf)
# best_estimators = ndms.take_best_estimators(mean_perf)

# # print(res2)
# ensamble = ndms.ensamble_estimators(best_estimators)
# # print(ensamble)

# train_test_data = bitcoin.cutting_slices("Close", "Close", len_data=45, times=1, offset_X=1,offset_y=0)

# result = ndms.train_model(ensamble, train_test_data)

# print(result)

# print(data)
# print(len(all_estimator))


# mean_performance = ndms.mean_performance_estimators(performance)

# print("Max :", sum(list(map(lambda x : x["Max_reference"], mean_performance))))

# X_train, X_test, y_train, y_test, date_pred = train_test_datas[0]

# data_prophet = pd.DataFrame()
# data_prophet["ds"] = X_train
# data_prophet["y"] = y_train

# prophet = Prophet()

# model = prophet.fit(data_prophet)

# future = model.make_future_dataframe(periods=1)

# forecast = prophet.predict(future)

# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# # view = View()
# # view.show_mean_estimators(estimators_and_results, each_results=True)

# best_estimators = ndms.take_best_estimators(estimators_and_results, 3)

# print(best_estimators)

# X, pred_date = bitcoin.data_to_predict(363, "Close","Date")
# print(X)
# print(ndms.make_prediction(best_estimators[0][0], X, pred_date))
# decision = ndms.take_decision(result)
# print(decision)

# estimators_and_results = ndms.train_model([MLPRegressor(**params)], train_test_datas)


# view = View()
# view.show_mean_estimators(estimators_and_results, each_results=True)

# train_test_data = bitcoin.cutting_slices("Close", "Close", len_data=600, times=1, offset_X=2,offset_y=1)
# Xtrain, Ytrain, Xtest, Ytest, pDate = train_test_data[0]

# day_data = bitcoin.cutting_slices("Close", "Close", len_data=1, times=1, offset_X=1,offset_y=0)
# _, _, Xtest_day, Ytest_day, pDate_day = day_data[0]

# model = Sequential()
# model.add(LSTM(45, return_sequences=True, input_shape=(600, 1))) 
# model.add(LSTM(90, return_sequences=True)) 
# # model.add(LSTM(45))
# # model.add(Dropout(0.1))
# model.add(Dense(1))

# model.compile(optimizer='adam', loss='mse')

# model.summary()

# result = model.fit(Xtrain.values.reshape(1, -1), Ytrain.values.ravel(),
#                    validation_data=(Xtest.values.reshape(1, -1), Ytest.values.ravel()),
#                    epochs=90, batch_size=45)

# test_loss, test_acc = model.evaluate(Xtest, Ytest, verbose=2)

# print("Acurácia: ", test_acc)
# plt.plot(result.history['loss'], label='Training loss')
# plt.plot(result.history['val_loss'], label='Validation loss')
# plt.legend()
# plt.show()

# predict = result.predict(Xtest)
# print("Predicao: ", predict[0][0], " Valor real: ",Ytest)
