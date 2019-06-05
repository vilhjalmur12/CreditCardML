from sklearn.neighbors import KNeighborsClassifier
from Service.Reporter import Reporter

class K_NearestNeighbour:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.set_classifier()
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.trainModel(self.X_train, self.Y_train)

    def set_classifier(self):
        self.classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

    def trainModel(self, X_train, Y_train):
        self.classifier.fit(X_train, Y_train)

    def predict(self, X_predict):
        return self.classifier.predict(X_predict)

    def report(self):
        Y_pred = self.predict(self.X_test)
        Reporter.report(self.classifier, self.X_train, self.Y_train, Y_pred, self.Y_test)




def KNN_classifier(X_train, Y_train, X_test, Y_test):
    # K-nearest neighbour classifier using euclidean distance metric
    from sklearn.neighbors import KNeighborsClassifier
    KNN_classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    KNN_classifier.fit(X_train,Y_train)
    Y_pred = KNN_classifier.predict(X_test)

    #reporter(KNN_classifier, X_train, Y_train, Y_pred, Y_test)

    #return KNN_classifier