from sklearn.neural_network import MLPClassifier
from Service.Reporter import Reporter

class Neural_Network_Classifier:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.set_classifier()
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.trainModel(self.X_train, self.Y_train)

    def set_classifier(self):
        self.classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    def trainModel(self, X_train, Y_train):
        self.classifier.fit(X_train, Y_train)

    def predict(self, X_predict):
        return self.classifier.predict(X_predict)

    def report(self):
        Y_pred = self.predict(self.X_test)
        Reporter.report(self.classifier, self.X_train, self.Y_train, Y_pred, self.Y_test)

