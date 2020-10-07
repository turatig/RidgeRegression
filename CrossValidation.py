"""
 This module provides functions to perform cross-validated statistical risk estimates
 """
import numpy as np
import itertools as it
from RidgeRegression import RidgeRegression
from Metrics import *
from Preprocessing import Pipe

#### Yield k test folds and the corresponding train part one by one
def kFoldIterator(X,y,k):
    #### (X.shape[0]+k-1)//k : add k-1 to handle train set which size m is not divisible for k
    m=X.shape[0]+k-1

    for i in range(k):
        #### Train part of the k-fold
        trainX=np.concatenate( [X[ (i-1)*(m//k) : i*(m//k) ], 
                                X[ (i+1)*(m//k) : ]] )

        trainY=np.concatenate( [y[ (i-1)*(m//k) : i*(m//k) ], 
                                y[ (i+1)*(m//k) : ]] )
        
        #### Test part of the k-fold
        testX=X[ i*(m//k) : (i+1)*(m//k) ]
        testY=y[ i*(m//k) : (i+1)*(m//k) ]

        yield {"train": {"data":trainX,"target":trainY}, 
                "test": {"data":testX,"target":testY}
            }
            
"""
  Compute a cross-validated estimate
"""           
def CVEstimate(estimator,X,y,k=5,metric=mse):

    testErr=[]
    trainErr=[]
    for fold in kFoldIterator(X,y,k):
        estimator=estimator.fit(fold["train"]["data"],fold["train"]["target"])

        trainErr.append( metric(estimator.predict(fold["train"]["data"]),
                            fold["train"]["target"]) )
        testErr.append( metric(estimator.predict(fold["test"]["data"]),
                            fold["test"]["target"]) )

    #### Computing mean test error and variance of the predictors
    mean=1/k*sum(testErr)
    meanTrain=1/k*sum(trainErr)
    return mean,meanTrain
    



"""
    Perform a grid search cross validation.
    Return a listed of scores sorted by meanScore in ascending order
"""
@taketime
def GridSearchCV(estimator,hparams,X,y,k=5,metric=mse):
    
    scoresList=[]
    m=X.shape[0]

    #### Testing on any combination of the params
    for combination in list(it.product(*hparams.values())):

        #### Setting the hyperparam of the algorithm
        h={d[0] : d[1] for d in zip(hparams.keys(),combination)}
        
        estimator=estimator.copy()
        estimator.set_params(**h)

        mean,meanTrain=CVEstimate(estimator,X,y,k,metric)

        estimator=estimator.fit(X,y)

        #### If the type of the estimator is a pipeline, keep only data referred to estimator
        if(type(estimator) is Pipe):
            estimator=estimator.estimator
        scoresList.append({"estimator": estimator,"meanScore":mean,"meanTrainScore":meanTrain})

    scoresList.sort(key=lambda e:e["meanScore"])
    return scoresList
            
#### Perform nested cross validation estimate
@taketime
def NestedCVEstimate(estimator,hparams,X,y,k,metric="mse"):
    estimatedRisk=[]
    m=X.shape[0]

    for fold in kFoldIterator(X,y,k):
        #### Grid search to find best hyperParams for test fold
        scoresList=GridSearchCV(estimator,hparams,fold["test"]["data"],fold["test"]["target"],k)

        """Train the algorithm on the train part of the fold with the best hyperParams 
            found with internal CV
        """
        estimator.set_params(**scoresList[0]["estimator"].get_params())
        estimator=estimator.fit(fold["train"]["data"],fold["train"]["target"])

        estimatedRisk.append( metric (estimator.predict(fold["test"]["data"]),
                            fold["test"]["target"]))

    return (1/k)*sum(estimatedRisk)

    

