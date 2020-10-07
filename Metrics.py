"""
This module provides functions for the evaluation of a predictor
"""
import numpy as np
import time


#### Functions to evaluate the performance of a predictor
#### yp: predicted labels
#### y: target labels

#### Mean squared error
def mse(yp,y): return np.average((yp-y)**2,axis=0)

#### Accuracy
def r2(yp,y): return 1-sum( (y-yp)**2 )/sum((y-np.average(y))**2)

#### Decorator to log the time of a computation
def taketime(f):
    def _wrap(*args,**kwargs):
        start=time.time()
        res=f(*args,**kwargs)
        end=time.time()
        print("\n","-"*100,"\n")
        print("Function {0} finished in time {1}".format(f.__name__,end-start))
        print("\n","-"*100)
        return res
    return _wrap

