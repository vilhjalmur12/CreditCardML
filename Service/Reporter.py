from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np

class Reporter:
    def report(self, classifier, X_train, Y_train, Y_pred, Y_test):

        cm = confusion_matrix(Y_test, Y_pred)

        print("The confusion matrix")
        print(cm)
        print()

        cl_report = metrics.classification_report(Y_test, Y_pred)


        print("Classifier report \n %s\n"
              % (metrics.classification_report(Y_test, Y_pred)))
        print()

        # K-Fold cross validation

        accuracies = cross_val_score(estimator=classifier, X=X_train,
                                     y=Y_train, cv=10)

        print("Accuracies")
        print(accuracies.mean())


        flatten_cm = []
        for column in cm:
            flatten_cm.append([np.int16(row).item() for row in column])

        flatten_accuracies = [np.float32(col).item() for col in accuracies]

        return { 'confusion_matrix': flatten_cm, 'report': cl_report,
                 'cross_validation': flatten_accuracies, 'cross_validation_mean': accuracies.mean()}