import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import * 

class Data_Model:
    
    data_frame = pd.DataFrame()
    data_frame_offset = pd.DataFrame()
    
    def __init__(self, symbol_1="BTC-USD", interval = "1d", period = "2y", start = None, end = None):
        self.data_frame = yf.Ticker(symbol_1).history(interval=interval, period=period, start=start, end=end)
        self.data_frame.reset_index(inplace=True)
    
    # Faz o processamento do dataFrame para recorte posterior
    def processing_data(self, column, name_column):
        self.data_frame[name_column] = self.data_frame[column].pct_change()
        self.data_frame = self.data_frame.iloc[1:]
        self.data_frame = self.data_frame.drop(["Stock Splits", "Dividends"], axis=1)
        self.data_frame['Mean HL'] = (self.data_frame['High'] + self.data_frame['Low']) / 2
        self.data_frame['Mean HLC'] = (self.data_frame['High'] + self.data_frame['Low'] + self.data_frame['Close']) / 3
        print(self.data_frame)
    
    def filter_data(self, column):
        scaler = StandardScaler()
        self.data_frame[column] = scaler.fit_transform(self.data_frame[column].values.reshape(-1, 1))
        print(self.data_frame)
        
        return scaler
    
    # Usa o data_frame para criar uma base de treinamento para os estimadores
    # retorna uma tupla(X_train, y_train)
    def split_data(self, columns = ("Close", "Close", "Date"), offset = 20, total_len = 360, X_y_len = (45, 1)):
        
        X_column, y_column, date_column    = columns
        X_len, y_len                = X_y_len # X_len Periodo para predição (in/entrada), y_len peiodo previsto (out/resultado/saída)
        
        end     = self.data_frame.last_valid_index() - offset
        start   = end - total_len
        
        data_X = self.data_frame[X_column]
        data_y = self.data_frame[y_column]
        data_date = self.data_frame[date_column]
        
        X_array = list()
        y_array = list()
        date_array = list()
        
        # X_len + y_len = período para predição (x) + periodo previsto previstos (y) de uma base de treinamento
        for i in range(start, end - X_len - y_len + 1):
            X = data_X.loc[data_X.index > i]
            X = X.loc[X.index < i + X_len + y_len]
            X_array.append(X)
            
            y = data_y.loc[data_y.index == i + X_len + y_len]
            y_array.append(y)
            
            date = data_date.loc[y.index]
            date = date.tolist()[0]
            date_array.append(date)
            
        X_train = np.array(X_array)
        y_train = np.array(y_array)
        
        return X_train, y_train, date_array
    
    # Retorna uma base para treinamento e n bases para teste
    def split_train_test(self, amount_tests = 20, train_len = 45, X_y_len = (7, 1)
                            , columns = ("Volatil", "Volatil", "Date")):
        
        test_len = sum(X_y_len)
        
        # Separando treinamento
        X_train, y_train, _ = self.split_data(columns, amount_tests, train_len, X_y_len)
        
        tests_array = list()
        
        # Separando testes
        for i in range(0, amount_tests):
            
            test = self.split_data(columns, i, test_len, X_y_len)
            tests_array.append(test)
        
        # Salvando em um dictionary
        train_tests = { 'X_train'   : X_train,
                        'y_train'   : y_train,
                        'Tests'     : tests_array}
        
        return train_tests
    
    def data_to_predict(self, index ,X_column, id_column = "Date"):
        
        return (self.data_frame[X_column]).loc[self.data_frame.index == index], (self.data_frame[id_column]).loc[self.data_frame.index == index]



