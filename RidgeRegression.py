"""
This module contains the class that implements the Ridge Regression algorithm
"""
import numpy as np
import Preprocessing as pr

class RidgeRegression:

    """   
        Fields:
        alpha : stabilization hyperparameter
        w : linear predictor belonging to R^d
        fit_intercept: if True, fit the intercept as the mean of y during fit
    """
    def __init__(self,alpha=1,fit_intercept=True):
        
        self.alpha=alpha
        self.fit_intercept=fit_intercept
        self.w=np.array([])
        self.intercept=0.

    def getAlpha(self): return self.alpha
    def getFitIntercept(self): return self.fit_intercept
    def getIntercept(self): return self.intercept
    def getCoefs(self): return self.w
    def getFitIntercept(self): return self.fit_intercept

    def setAlpha(self,a): 
        if a>0: self.alpha=a
        
    def setFitIntercept(self,a):self.fit_intercept=a

    #### This methods implement the generic interface to get/set the hyperparameters of the estimator.

    def get_params(self):
        return {
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept
        }

    def set_params(self,**d):
        _d={"alpha": lambda a: self.setAlpha(a),
            "fit_intercept": lambda a: self.setFitIntercept(a)
            }

        for k,v in d.items():
            try: _d[k](v)
            except KeyError as e: print("The algorithm doesn't have {0} as hyperparameter".format(k))
    
    #### Return a new object of this class set with the same params of the caller
    def copy(self):
        return RidgeRegression(**self.get_params())
    """
        Compute w which optimize || Xw - y ||^2 + alpha || w ||^2
        X : mxd design matrix where m=trainset_size d=feature_space_size
        y : labels vector belonging to R^m
    """

    def fit(self,X,y):
        
        if self.fit_intercept:
            #### Compute the mean value of every feature in train set
            X_offset=np.average(X,axis=0)
            y_offset=np.average(y,axis=0)
            X=pr.center(X)
            
        #### w = (X^T X + alpha I) ^ (-1) + X^T y  
        invertible_mat=np.matmul(X.T,X) + self.alpha * np.identity(X.shape[1])
        self.w= np.matmul( np.linalg.inv( invertible_mat ), np.matmul( X.T, y) )

        if self.fit_intercept:
            #### The predictor will output the y centroid given the X centroid
            self.intercept=y_offset-np.dot(X_offset,self.w)
        return self

    #### Return a list of predicted labels for a set of unseen data points.
    def predict(self,X):
        return np.matmul(X,self.w)+self.intercept

    #### Return a list of loss function(y,y_predicted) 
    def test(self,X,y):
        return (y-self.predict(X))**2

    def __str__(self):
        return "alpha: {0}\nfit_intercept: {1}\nw: {2}\n".format(self.alpha,self.fit_intercept,self.w)

