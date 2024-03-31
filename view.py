from includes import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class View:
    
    def __init__(self):
        pass
        
    def ShowResults(self, results):
        # Mostra os resultados de todos os modelos treinados
        for estimator_result in results:
            print(f"Estimator: {estimator_result['Estimator']}")
            for result in estimator_result['Results']:
                date = result['Date']
                predicted_value = result['Predict']
                real_value = result['Real_value']
                relative_error = result['Relative_error']

                print(f"Date: {date}, Real: {real_value}, Predicted: {predicted_value}, Error: {relative_error:.4%}")
    
    def ShowComparisons(self, results):
        # Mostra a comparação dos resultados
        
        pass
    
    def show_graphics(self, estimators_and_results):
        i = 1
        for estimator in estimators_and_results:
            rows = cols = m.ceil(len(estimators_and_results)**(1/2))
            plt.subplot(rows,cols,i)

            y = list(map(lambda x : x['Date'], estimator['Results']))
            x = list(map(lambda x : x['Predict'], estimator['Results']))
            x = np.array(x).reshape(-1, 1)
            plt.plot(y, x, label = 'Previsto')
            x = list(map(lambda x : x['Real_value'], estimator['Results']))
            x = np.array(x).reshape(-1, 1)
            plt.plot(y, x, label ='Real')
            
            plt.xticks(rotation=20, fontsize=8)
            plt.yticks(fontsize=8)
            
            plt.title(str(estimator['Estimator']), fontsize=10)
            plt.legend()
            plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95, wspace=0.2, hspace=0.35)
            i += 1
        
        plt.show()
        
        