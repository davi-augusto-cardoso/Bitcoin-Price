from datetime import timedelta
import statistics
import pandas as pd
from storage import Data_Model
from functools import partial
from skopt import dummy_minimize, gp_minimize

from estimators_utils import *

class Nostradamus:

    def __init__(self, data_model):
        self.data_model = data_model
        # self.data_model.processing_data()
            
    # retorna Lista[Dicionário{Estimador, DataFrame]}] 
    def train_tunning_model(self, estimators, train_test_datas, scaler):
        
        estimators_predictions = list() # Dicionário{Estimador, Lista[Dicionário{data, valor previsto, valor real}]}

        # Treinando estimador por estimador com o período temporal escolhido
        for estimator in estimators:

            results = list()
            
            X_train = train_test_datas['X_train']
            y_train = train_test_datas['y_train']
            
            # Escolhendo melhores parametros para o período temporal
            model = self.tunning_model( estimator = estimator['Estimator'], params_names = estimator['Params_names'],  
                            train_data = (X_train, y_train), space = estimator['Space'])
            
            for test_data in train_test_datas['Tests']:
                
                X_test, y_test, date = test_data
                
                prediction = model.predict(X_test)
                
                prediction = scaler.inverse_transform(prediction.reshape(-1, 1))[0]
                
                try:
                    target = scaler.inverse_transform(y_test.tolist())[0]
                except:
                    target = 1
                    
                if prediction <= 1:
                    relative_error = prediction + target
                else:
                    relative_error = abs(target-prediction[0])/target
                    
                result = {
                    'Date'              : date[0],
                    'Real_value'        : target,
                    'Predict'           : prediction,
                    'Relative_error'    : relative_error[0],
                    'Yesterday_val'     : scaler.inverse_transform(X_test.reshape(-1, 1))[-1][-1],
                    'Best_model'        : model
                }
                
                results.append(result)
            
            estimator_predictions = {
                "Estimator" : estimator['Estimator'],
                "Results"   : results
            }
            
            estimators_predictions.append(estimator_predictions)
            
        return estimators_predictions
    
    # Treina e testa um grupo de algorítimos estiamdores
    # retorna Lista[Dicionário{Estimador, DataFrame]}] 
    def train_model(self, estimators, train_test_datas):
        
        estimators_predictions = list() # Dicionário{Estimador, Lista[Dicionário{data, valor previsto, valor real}]}

        for estimator in estimators:

            results = list()
            
            X_train = train_test_datas['X_train']
            y_train = train_test_datas['y_train']
            
            # Treinando modelo
            model = estimator.fit(  X = X_train, y = y_train.ravel())
            
            for test_data in train_test_datas['Tests']:
                
                X_test, y_test, date = test_data
                
                prediction = model.predict(X_test)

                try:
                    target = y_test.tolist()[0]
                except:
                    target = 1
                    
                if prediction[0] <= 1:
                    relative_error = prediction[0] + target
                else:
                    relative_error = abs(target-prediction[0])/target
                    
                result = {
                    'Date'              : date[0],
                    'Real_value'        : target,
                    'Predict'           : prediction,
                    'Relative_error'    : relative_error[0]
                }
                
                results.append(result)
            
            estimator_predictions = {
                "Estimator" : estimator,
                "Results"   : results
            }
            
            estimators_predictions.append(estimator_predictions)
            
        return estimators_predictions

    # Retorna statisticas das n predições (Média, desvio, erro relativo ...)
    # Performance individual dos estimadores
    def performance_individual(self, estimators_predictions):
        
        performances = list()
        
        for estimator in estimators_predictions:
            
            predictions     = list(map(lambda x : x['Predict'], estimator['Results']))
            relative_errors = list(map(lambda x : x['Relative_error'], estimator['Results']))
            
            max_relative_error          = max(relative_errors)
            mean_relative_error         = statistics.mean(relative_errors)
            
            try:
                deviation_relative_error    = statistics.stdev(relative_errors)
                quantiles                   = statistics.quantiles(relative_errors, n=4)
            except:
                deviation_relative_error    = 0
                quantiles                   = list()

            performance = {
                'Estimator'                 : estimator['Estimator'],
                'Max_relative_error'        : max_relative_error,
                'Mean_relative_error'       : mean_relative_error,
                'Deviation_relative_error'  : deviation_relative_error,
                'Quantiles'                 : quantiles,
                'Amount_quantile'           : 0,
                'Note'                      : 0,
                'Relative_errors'           : relative_errors,  
                'Predictions'               : predictions,             
            }
            performances.append(performance)
        
        # performance_individual = self.mean_performance_estimators(performances)
        
        return performances
    
    # Performance geral dos estimadores
    def performance_general(self, performance_individual, amount_estimators=5, quantile = 1):
        
        estimators              = sorted(performance_individual, key=lambda x : x['Max_relative_error' ])[:amount_estimators]

        max_reference           = statistics.mean(list(map(lambda x : x['Max_relative_error'], estimators)))
        reference_means         = statistics.mean(list(map(lambda x : x['Mean_relative_error'], estimators))) 
        reference_deviations    = statistics.mean(list(map(lambda x : x['Deviation_relative_error'], estimators))) 
        try:
            reference_quantiles = statistics.mean(list(map(lambda x : x['Quantiles'][quantile], estimators)))
        except:
            reference_quantiles = list()
        
        reference_estimators = {
            "Estimators"            : estimators,
            "Max_reference"         : max_reference,
            "Mean_reference"        : reference_means,
            "Deviation_reference"   : reference_deviations,
            "Quantile_reference"    : reference_quantiles,
        }
        
        return reference_estimators
    
    def take_best_estimators(self, reference, number_estimators = 2, weights_qrt=[2, 1, 1, 0, 0], quantile = 1):

        for estimator in reference['Estimators']:
            estimator['Note'] += 1 if estimator['Mean_relative_error'] <=  reference['Mean_reference'] else 0
            estimator['Note'] += 1 if estimator['Deviation_relative_error'] <=  reference['Deviation_reference'] else 0
            try:
                estimator['Note'] += 1 if estimator['Quantiles'][quantile] <=  reference['Quantile_reference'] else 0
                estimator['Amount_quantile'] = sum(list(map(lambda x : 1 if x <= reference['Quantile_reference'] else 0, estimator['Relative_errors'])))
            except:
                pass
        try:
            reference['Estimators'].sort(key=lambda x : x['Amount_quantile'])
            
            for i in range(len(reference['Estimators'])-1, 0):
                reference['Estimator'][i]['Note'] += weights_qrt[i]
        except:
            pass
        
        reference['Estimators'].sort(key=lambda x : x['Note'])

        best_estimators = [ref['Estimator'] for ref in reference['Estimators'][len(reference['Estimators'])-number_estimators:]]
        
        return best_estimators

    def ensamble_estimators(self, estimators):
        
        estimators_list = list()
        for i in range(0, len(estimators)):
            new_tuple = ("Estimators_"+str(i), estimators[i])
            estimators_list.append(new_tuple)
            
        return [(VotingRegressor(estimators=estimators_list))]

    def testing_ensambles(self, X_column="Close", y_column="Close", estimators= best_estimators,
                          times_ensambles=20, times_estimators=20):
        ensamble_res = list()
        
        for i in range(times_ensambles+1, 1, -1):
            train_test_datas = self.data_model.cutting_slices(X_column, y_column, len_data=45, times=times_estimators, offset_X=i-2,offset_y=i-1)

            estimators_and_results = self.train_model(estimators, train_test_datas)

            performance = self.performance_general(estimators_and_results)

            best_estimators = self.take_best_estimators(performance)

            ensamble = self.ensamble_estimators(best_estimators)

            train_test_data = self.data_model.cutting_slices(X_column, y_column, len_data=45, times=1, offset_X=i-1,offset_y=i)

            estimator_and_result = self.train_model(ensamble, train_test_data)
            
            ensamble_res.append(estimator_and_result[0])

        performance = self.performance_general(ensamble_res)

        print(performance)
        # return performance

    def make_estimators_data(self, X_column="Close", y_column="Close", len_data_frame=10, n_days=45):
        
        Days = list(list() for i in range(0, n_days))
        Estimators = list()
        
        for i in range(1, len_data_frame):
            train_test_data = self.data_model.cutting_slices(X_column, y_column, len_data=n_days, times=1, offset_X=i,offset_y=i-1)
            estimators_and_results = self.train_model(all_estimator, train_test_data)
            dayli_performance = self.performance_estimators(estimators_and_results)
            dayli_performance.sort(key= lambda x : x["Max_relative_error"])

            for j in range(0, 45):
                Days[j].append(train_test_data[0][0].iloc[j])   
                                    
            Estimators.append(dayli_performance[0]["Estimator"])
            
        data = {}
        for i in range(0, n_days):
            data["Day_" + str(i)] = Days[i]
            
        data["Estimator"] = Estimators
        data_frame = pd.DataFrame(data)

        return data_frame
        
    def tunner(self, space, estimator, params_names, train_data):
        X_train, y_train = train_data
        
        params = dict(zip(params_names, space))
        model  = estimator(**params)
        
        model.fit( X = X_train, y = y_train)
        
        score = model.score(X = X_train, y = y_train)
        
        return -score
    
    def tunning_model(self, estimator, params_names, train_data, space):
        
        optimal_func = partial(
            self.tunner,
            estimator       = estimator,
            params_names    = params_names,
            train_data      = train_data)

        best_params = dummy_minimize(optimal_func, 
                                dimensions = space, 
                                random_state = 20,
                                n_calls=60, verbose=False)

        params = dict(zip(params_names, best_params.x))
        
        X_train, y_train = train_data
        
        estimator = estimator(**params)
        
        model_parametrized = estimator.fit(X_train, y_train)
        
        return model_parametrized
    
    # Mensurando a lucratividade do modelo
    def profit_model(self, estimators_predictions, value=100):
        
        for estimator in estimators_predictions:
            
            print('Estimator: ', estimator['Estimator'])
            wallet = value
            bitcoin = 0
            
            for result in estimator['Results']:
                
                if(result['Yesterday_val'] - result['Predict'] > 0):
                    
                    if(wallet > bitcoin):
                        # Compra
                        bitcoin = wallet
                        wallet  = 0
                    
                    gain = (result['Real_value'] - result['Yesterday_val'])/result['Yesterday_val']
                    
                    bitcoin += (gain * bitcoin)
                else:
                    if(bitcoin > wallet):
                        # Venda
                        wallet  = bitcoin
                        bitcoin = 0
                
            if(bitcoin > wallet):
                wallet = bitcoin
            
            print('Profit: ', wallet[0], wallet[0]-100, "%")