# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 10:40:47 2017

@author: betienne
"""

import numpy as np
import time
from scipy.optimize import fmin_l_bfgs_b


class HoltWinters():
    """
    Implementation of Holt-Winters exponential smoothing

    """
    def __init__(self, method, m=None):
        """
        Initiates a HoltWinters smoothing

        Parameters:
            method : additive or multiplicative
            m : seasonality
        """
        self.method = method
        self.m = m
        if self.method == 'm':
            print("\t Multiplicative Model - Seasonality : {}".format(self.m))
        elif self.method == 'a':
            print("\t Additive Model - Seasonality : {}".format(self.m))
        elif self.method =='l':
            print("\t Linear Model")
        else:
            print("Method not valid. Default will be Linear")


    def fit(self, X, metric='MAPE'):
        """
        Fits a model

        Parameters:
            X : time-series
            metric : MAPE or RMSE
        """
        self.X = X
        self.y = np.array(self.X[:].reshape(-1))
        self.x0 = np.random.rand(3)
        self.bounds = [(0, 1),(0, 1),(0, 1)]
        self.metric = metric
        print('*'*50)
        print('Looking for optimal values of alpha, beta, gamma', end="...")
        print("\nInitial Values x0: {} // Seasonality model : {}".format(self.x0, self.method))

        try:
            self.t0 = time.clock()
            if self.metric == 'RMSE':
                self.x = fmin_l_bfgs_b(RMSE, x0 = self.x0,
                                       args = (self.y, self.m, self.method),
                                       approx_grad=True,
                                       bounds = self.bounds)
            else:
                self.x = fmin_l_bfgs_b(MAPE, x0 = self.x0,
                                       args = (self.y, self.m, self.method),
                                       approx_grad=True,
                                       bounds = self.bounds)
            print("\nalpha = {}".format(self.x[0][0]))
            print("beta = {}".format(self.x[0][1]))
            print("gamma = {}".format(self.x[0][2]))
            print("Value at local minimum = {}".format(self.x[1]))
            for k ,v in self.x[2].items():
                print("{} : {}".format(k, v))

        except Exception as e:

            print("Erreur : {}".format(e))
            print("Values of alpha, beta, gamma were not set")
            print('*'*50)

        finally:

            print('Time elapsed : {}'.format(time.clock() - self.t0))
            print('*'*50)
            self.alpha, self.beta, self.gamma = self.x[0]


    def _multiplicative(self):

    #Initialisation
        self.l = np.array([np.mean(self.y[:self.m])])
        self.b = np.array([(1. / (self.m)**2) * (np.sum(self.y[self.m:(2*self.m)]) - np.sum(self.y[:self.m]))])
        self.s = np.array([self.y[:self.m] / self.l[0]]).T.reshape(-1)
        self.y_hat = np.array([])
    #Itérations
        for i in range(len(self.y)):
            self.l = np.append(self.l, self.alpha*(self.y[i] / self.s[-self.m]) + (1.-self.alpha)*(self.l[-1] + self.b[-1]))
            self.b = np.append(self.b, self.beta*(self.l[-1] - self.l[-2]) + (1.-self.beta)*self.b[-1])
            self.s = np.append(self.s, self.gamma * (self.y[i] / (self.l[-1] + self.b[-1])) + (1.-self.gamma)*self.s[-self.m])
            self.y_hat = np.append(self.y_hat, (self.l[-1] + self.b[-1]) * self.s[-1])
#            print("y: {:2f}, l: {:2f}, b: {:2f}, s: {:2f}, y_hat: {:2f}".format(self.y[i], self.l[-1], self.b[-1], self.s[-1], self.y_hat[-1]))


    def _additive(self):

    #Initialisation
        self.l = np.array([np.mean(self.y[:self.m])])
        self.b = np.array([(1. / (self.m)**2) * (np.sum(self.y[self.m:(2*self.m)]) - np.sum(self.y[:self.m]))])
        self.s = np.array([self.y[:self.m] - self.l[0]]).T.reshape(-1)
        self.y_hat = np.array([])
    #Itérations
        for i in range(len(self.y)):
            self.l = np.append(self.l, self.alpha*(self.y[i] - self.s[-self.m]) + (1.-self.alpha)*(self.l[-1] + self.b[-1]))
            self.b = np.append(self.b, self.beta*(self.l[-1] - self.l[-2]) + (1.-self.beta)*self.b[-1])
            self.s = np.append(self.s, self.gamma * (self.y[i] - (self.l[-1] + self.b[-1])) + (1.-self.gamma)*self.s[-self.m])
            self.y_hat = np.append(self.y_hat, (self.l[-1] + self.b[-1]) * self.s[-1])
#            print("y: {:2f}, l: {:2f}, b: {:2f}, s: {:2f}, y_hat: {:2f}".format(self.y[i], self.l[-1], self.b[-1], self.s[-1], self.y_hat[-1]))


    def _linear(self):

    #Initialisation
        self.l = np.array([self.y[0]])
        self.b = np.array([self.y[1] - self.y[0]])
        self.y_hat = np.array([])
    #Itérations
        for i in range(len(self.y)):
            self.l = np.append(self.l, self.alpha*(self.y[i]) + (1.-self.alpha)*(self.l[-1] + self.b[-1]))
            self.b = np.append(self.b, self.beta*(self.l[-1] - self.l[-2]) + (1.-self.beta)*self.b[-1])
            self.y_hat = np.append(self.y_hat, (self.l[-1] + self.b[-1]))


    def predict(self, h):
        """
            Predicts the values of a fitted HW model

        Parameters :
            h : forecast horizon
        """
        self.h = h
        self.y_hat = np.array([])

        if self.method == 'm':
            self._multiplicative()
            for i in range(1, self.h+1):
                self.y_hat = np.append(self.y_hat,(self.l[-1] + i*self.b[-1])*self.s[i%self.m])

        elif self.method == 'a':
            self._additive()
            for i in range(1, self.h+1):
                self.y_hat = np.append(self.y_hat,(self.l[-1] + i*self.b[-1]) + self.s[i%self.m])

        else:
            self._linear()
            for i in range(1, self.h+1):
                self.y_hat = np.append(self.y_hat,(self.l[-1] + i*self.b[-1]))

        return(self.y_hat[-self.h:])
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def MAPE(params, *args):
    """
    Mean Average Percentage Error

    """
    alpha, beta, gamma = params
    X = args[0]
    m = args[1]
    method = args[2]

    if m == None:
        y = np.array(X[:].reshape(-1))
        l_0 = y[0]
        b_0 = y[1] - y[0]
        l = np.array([l_0])
        b = np.array([b_0])

        y_hat = np.array([])

        for i in range(len(y)):

            l = np.append(l, alpha*(y[i]) + (1.-alpha)*(l[i] + b[i]))
            b = np.append(b, beta*(l[i+1] - l[i]) + (1.-beta)*b[i])
            y_hat = np.append(y_hat, (l[i] + b[i]))

    else:
        y = np.array(X[:].reshape(-1))
        l_0 = np.mean(y[:m])
        b_0 = (np.sum(y[m : (2*m)]) - np.sum(y[:m])) / (m**2)
        l = np.array([l_0])
        b = np.array([b_0])

        if method == "m":

            s = np.array([y[:m] / l[0]]).T.reshape(-1)
            y_hat = np.array([])

            for i in range(len(y)):

                l = np.append(l, alpha*(y[i] / s[i]) + (1.-alpha)*(l[i] + b[i]))
                b = np.append(b, beta*(l[i+1] - l[i]) + (1.-beta)*b[i])
                s = np.append(s, gamma * (y[i] / (l[i] + b[i])) + (1.-gamma)*s[i])
                y_hat = np.append(y_hat, (l[i] + b[i]) * s[i])

        else:
            s = np.array([y[:m] - l[0]]).T.reshape(-1)
            y_hat = np.array([])

            for i in range(len(y)):

                l = np.append(l, alpha*(y[i] - s[i]) + (1.-alpha)*(l[i] + b[i]))
                b = np.append(b, beta*(l[i+1] - l[i]) + (1.-beta)*b[i])
                s = np.append(s, gamma * (y[i] - (l[i] + b[i])) + (1.-gamma)*s[i])
                y_hat = np.append(y_hat, (l[i] + b[i]) + s[i])

    return(np.mean(np.sum(np.abs((y - y_hat)/y))))
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def RMSE(params, *args):
    """
    Root Mean Squared Error

    """

    alpha, beta, gamma = params
    X = args[0]
    m = args[1]
    method = args[2]

    if m == None:
        y = np.array(X[:].reshape(-1))
        l_0 = y[0]
        b_0 = y[1] - y[0]
        l = np.array([l_0])
        b = np.array([b_0])

        y_hat = np.array([])

        for i in range(len(y)):

            l = np.append(l, alpha*(y[i]) + (1.-alpha)*(l[i] + b[i]))
            b = np.append(b, beta*(l[i+1] - l[i]) + (1.-beta)*b[i])
            y_hat = np.append(y_hat, (l[i] + b[i]))

    else:
        y = np.array(X[:].reshape(-1))
        l_0 = np.mean(y[:m])
        b_0 = (np.sum(y[m : (2*m)]) - np.sum(y[:m])) / (m**2)
        l = np.array([l_0])
        b = np.array([b_0])

        if method == "m":

            s = np.array([y[:m] / l[0]]).T.reshape(-1)
            y_hat = np.array([])

            for i in range(len(y)):

                l = np.append(l, alpha*(y[i] / s[i]) + (1.-alpha)*(l[i] + b[i]))
                b = np.append(b, beta*(l[i+1] - l[i]) + (1.-beta)*b[i])
                s = np.append(s, gamma * (y[i] / (l[i] + b[i])) + (1.-gamma)*s[i])
                y_hat = np.append(y_hat, (l[i] + b[i]) * s[i])

        else:
            s = np.array([y[:m] - l[0]]).T.reshape(-1)
            y_hat = np.array([])

            for i in range(len(y)):

                l = np.append(l, alpha*(y[i] - s[i]) + (1.-alpha)*(l[i] + b[i]))
                b = np.append(b, beta*(l[i+1] - l[i]) + (1.-beta)*b[i])
                s = np.append(s, gamma * (y[i] - (l[i] + b[i])) + (1.-gamma)*s[i])
                y_hat = np.append(y_hat, (l[i] + b[i]) + s[i])

    return(np.mean(np.sum(np.power((y - y_hat),2))))