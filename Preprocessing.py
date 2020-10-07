"""
This module provides function to preprocess data
"""
import numpy as np
import numpy.random as rn

#### Center data points in mat A by subtracting the mean
def center(X): return X-np.average(X,axis=0)

#### Scale data points to have them with variance=1
def scale(X): return X/np.std(X,axis=0)

#### Return (X-Xm)/Xstd
def stdScale(X): return scale(center(X))

#### Util to shuffleDataset
def shuffleDataset(X,y):
    y=np.array([[e] for e in y])
    a=np.concatenate([X,y],axis=1)
    rn.shuffle(a)
    return np.array([row[:-1] for row in a]),np.array([row[-1] for row in a])


""" 
Cache results of transformation applied within the last n computations
"""
def cache_transformer(f):
    __CACHE__=[]
    def _wrap(self,X):
        nonlocal __CACHE__
        for i in __CACHE__:
            if(i["in"] is X):
                return i["out"]
        if len(__CACHE__)>=5:
            __CACHE__=__CACHE__[1:]
        __CACHE__.append({"in":X,"out":f(self,X)})
        return __CACHE__[-1]["out"]
    return _wrap

#### Class that implements a simple standard scaler
class StdScaler():

    def __init__(self,mean=0.,std_dev=0.):
        self.mean=mean
        self.std_dev=std_dev

    def fit(self,X):
        self.mean=np.average(X,axis=0)
        self.std_dev=np.std(X,axis=0)
        return self

    @cache_transformer
    def transform(self,X):
        #### Return transformed data and handle the case in which some features have std_dev=0 
        if len([el for el in self.std_dev if el==0])==0:
            return (X-self.mean)/self.std_dev
        else:
            return np.array([\
                        [ row[i]/self.std_dev[i] if self.std_dev[i]!=0 else row[i] for i in range(X.shape[1])]\
                        for row in X])

    def get_params(self):
        return {"mean":self.mean,"std_dev":self.std_dev}

#### Class to make a pipeline composed by a list of tansformers and an estimator
class Pipe():
    def __init__(self,transformers=[],estimator=None):
        #### List of transformer object
        self.transformers=[el for el in transformers]
        self.estimator=estimator

    def fit(self,X,y=None):
        for t in self.transformers:
            t.fit(X)
            X=t.transform(X)
        if(self.estimator is not None and y is not None):
            self.estimator.fit(X,y)
        return self

    ###Recursively apply transformations from first to last
    @cache_transformer
    def transform(self,X):
        def apply(t,_X):
            if not t: return _X 
            else: return apply(t[1:],t[0].transform(_X))
        return apply(self.transformers,X)

    def predict(self,X):
        if(self.estimator is not None):
            return self.estimator.predict(self.transform(X))
        return None

    def set_params(self,**d):
        self.estimator.set_params(**d)

    #### Return exact copy of a pipeline
    def copy(self):
        return Pipe([type(t)(**t.get_params()) for t in self.transformers],self.estimator.copy())




    