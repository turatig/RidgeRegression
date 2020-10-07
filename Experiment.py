"""
This module contains functions useful to analyze the Ridge Regression implementation on a given dataset
"""
import numpy as np
from Preprocessing import *
from CrossValidation import GridSearchCV,NestedCVEstimate, CVEstimate
from Plots import *
from RidgeRegression import RidgeRegression
from Metrics import *

"""
    Compute GridSearchCV and nested cross-validation estimate and set plot axes
    start,stop,n_values are the parameters of the grid. k=number of folds for 
    cross-validation.
    Return the scoresList of the GridSearchCV. Plot coeffs vs alpha and riskEstimate vs alpha.  
"""
def estimateRegression(X,y,start,stop,n_values,k=5,metric=mse):
    scoresList=GridSearchCV(RidgeRegression(),{"alpha": np.linspace(start,stop,n_values),
                                                "fit_intercept":[False,True]},
                                                X,y,k,metric)
    nest_est=[]
    print("\n","*"*100,"\n")
    print("The best value of alpha found in range {0}-{1} and score values:".format(start,stop))

    for key,value in scoresList[0].items():
        print("{0} : {1}".format(key,value))
    print("\n","*"*100)

    inter={
        True: "Fit intercept",
        False: "Don't fit intercept",
    }
    #### First with fit_intercept=True then with fit_intercept=False
    for t in inter.keys():

        fig,axs=plt.subplots(2)
        axs[0].set_title("Predictor coefficients - {0}".format(inter[t]))
        axs[1].set_title("Cross-validated estimate - {0}".format(inter[t]))
        fig.tight_layout(pad=0.8)

        #### Plotting alpha vs coefs/riskEst for any transformation on data
        a=[r for r in scoresList if r["estimator"].getFitIntercept()==t]
        plotCoef(axs[0],a)
        plotGridSearch(axs[1],a)

        #### Perform nested cross-validated estimate for the couple (fit_intercept,scale) with a grid centered on the best
        #### value discovered with the GridSearchCv
        a.sort(key=lambda e: e["meanScore"])
        ncv_alpha=np.linspace(a[0]["estimator"].getAlpha()-( (stop-start)/n_values )/2,
                            a[0]["estimator"].getAlpha()+( (stop-start)/n_values )/2,
                            n_values)

        res=NestedCVEstimate(RidgeRegression(),{"alpha":ncv_alpha,"fit_intercept":[t]},X,y,k,metric)
        print("\n","*"*100,"\n")
        print("Nested cross-validation estimate for a grid centered around alpha={0} fit_intercept={1}".\
                    format(a[0]["estimator"].getAlpha(),a[0]["estimator"].getFitIntercept()))
        print(res)
        print("\n","*"*100)
        nest_est.append((t,res))

    nest_est.sort(key=lambda el:el[1])
    print("The best nested CV estimate is obtained with fit_intercept={0}".format(nest_est[0][0]))
    print("The difference between the two is {0}".format(abs(nest_est[0][1]-nest_est[1][1])))

    return scoresList


"""
    Shuffle dataSet to test how this operation changes CVEstimates
"""
@taketime
def shuffledCVEstimate(estimator,X,y):

    estimates=[]
    for i in range(30):
        X,y=shuffleDataset(X,y)
        mean,_=CVEstimate(estimator,X,y)
        estimates.append(mean)
    return estimates