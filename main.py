# Version 1.0

# System imports
import warnings

# Nostradamus imports
from storage import Data_Model
from nostradamus import Nostradamus
from view import View

# Utils imports
from estimators_utils import *
from spaces_utils import *

warnings.filterwarnings("ignore")

bitcoin = Data_Model(symbol_1="BTC-USD", interval="1d", period="4y")

bitcoin.processing_data('Close', 'Volatil')
# scaler = bitcoin.filter_data('Volume')
scaler = bitcoin.filter_data('Close')

ndms = Nostradamus(bitcoin)

for i in range(120, 360, 60):
    for j in range(5, 14, 1):
        train_test_data = bitcoin.split_train_test(amount_tests = 30, train_len = i, X_y_len = (j, 1)
                                                , columns = ("Close", "Close", "Date"))

        # estimators_and_results = ndms.train_model(best_estimators, train_test_data)
        estimators_and_results = ndms.train_tunning_model(estimators_hp, train_test_data, scaler)

        print("\nTrain_len: ", i, "X_len: ", j)
        ndms.profit_model(estimators_and_results)

performance_i = ndms.performance_individual(estimators_and_results)
# performance_g = ndms.performance_general(performance_i, estimators_and_results)

view = View()
# view.ShowResults(estimators_and_results)
# view.ShowResultsIndividual(performance_i)

