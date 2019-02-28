import numpy as np


class GaussianNaiveBayes(object):
    '''
    A class that represents a Gaussian Naïve Bayes classifier
    '''

    def __init__(self):
        self.mu = None
        self.std = None

        self.number_of_classes = None

    def fit(self, X, y):
        '''
        Calculates the mean and standard deviation for 
        each class.

        Arguments:
            X: the features of the training data
            y: the labels of the training data
        '''
        classes = set(y)
        number_of_data_points = X.shape[0]
        number_of_features = X.shape[1]
        number_of_classes = len(classes)

        mu = np.zeros((number_of_classes, number_of_features))
        std = np.zeros((number_of_classes, number_of_features))
        prior = np.zeros((number_of_classes, 1))

        for i, c in enumerate(classes):
            class_indices = np.where(y == c)
            X_class = X[class_indices]
            mu_class = np.mean(X_class, axis=0)
            std_class = np.mean((X_class - mu_class)**2, axis=0)

            mu[i, :] = mu_class
            std[i, :] = std_class
            prior[i] = X_class.shape[0] / number_of_data_points

        self.mu = mu
        self.std = std
        self.prior = prior
        self.number_of_classes = number_of_classes

    def predict(self, X):
        '''
        Predictions the class for given features.

        Arguments:
            X: the features of the data points to be classified.

        Returns:
            predictions: the predictions of the model.
            probs: the probabilities assigned to each data points.
        '''
        probs = []
        predictions = []

        for x in X:
            prob = np.zeros(self.number_of_classes)
            for c in range(self.number_of_classes):
                power = -0.5 * \
                    ((x - self.mu[c, :]) / (self.std[c, :] + 10E-100))**2

                coefficient = - \
                    np.log(np.sqrt(2 * np.pi * self.std[c, :] + 10E-10))

                log_likelihood = np.sum(coefficient + power)
                log_prior = np.log(self.prior[c])
                prob[c] = log_prior + log_likelihood

            prediction = np.argmax(prob)

            probs.append(prob)
            predictions.append(prediction)

        predictions = np.array(predictions)

        return predictions, probs

    def accuracy(self, y, y_hat):
        '''
        Calculates the accuracy of the model

        Arguments: 
            y: the true labels of the data
            y_hat: the predicted labels of the data
        '''

        number_of_points = y.shape[0]

        ratio = sum(y == y_hat) / number_of_points

        return ratio
