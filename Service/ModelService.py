from Classifiers.SVM_Classifier import SVM_Classifier
from Data.preprocess import dataPreprocess, predDataProcess
import numpy as np

class ModelService:
    def __init__(self, chosen_classifier):
        self.ModelName = chosen_classifier
        self.trainingModules = dataPreprocess()
        self.Model = SVM_Classifier(self.trainingModules['X_train'], self.trainingModules['Y_train'],
                                    self.trainingModules['X_test'], self. trainingModules['Y_test'])


    def modelReport(self):
        return self.Model.report()


    def dataValidator(self, obj):
        if not ( obj['step'] or obj['age'] or obj['gender'] or obj['merchant'] or obj['category'] or obj['amount']):
            return False
        else:
            return True

    def predict(self, value_obj):
        values = value_obj['values']
        container = []

        try:
            for item in values:
                tmp_array = []
                tmp_array.append(float(item['step']))
                tmp_array.append(float(item['age']))
                tmp_array.append(float(item['gender']))
                tmp_array.append(float(item['merchant']))
                tmp_array.append(float(item['category']))
                tmp_array.append(float(item['amount']))
                container.append(tmp_array)
        except:
            return False

        X = predDataProcess(container)

        predi = self.Model.predict(X)
        print(predi)

        # return the prediction from set Model classifier
        return self.Model.predict(X).tolist()














