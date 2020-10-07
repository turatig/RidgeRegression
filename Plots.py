"""
This module provides plot functions
"""
import matplotlib.pyplot as plt
from copy import copy

"""
    All these functions take in argument the ax to plot onto and a scores list computed by GridSearchCV
"""
#### alpha vs predictor coefficients
def plotCoef(ax,scoresList):

    scoresList=copy(scoresList)
    scoresList.sort(key=lambda e:e["estimator"].getAlpha())

    ax.plot([ x["estimator"].getAlpha() for x in scoresList ],
            [ y["estimator"].getCoefs() for y in scoresList ])

    ax.set_xlabel("alpha")
    ax.set_ylabel("coeffs")

#### alpha vs CV risk estimate 
def plotTestErr(ax,scoresList):
    scoresList=copy(scoresList)
    scoresList.sort(key=lambda e:e["estimator"].getAlpha())

    ax.plot([ x["estimator"].getAlpha() for x in scoresList ],
            [ y["meanScore"] for y in scoresList ])

    ax.set_xlabel("alpha")
    ax.set_ylabel("test score")

#### Plot test score and train score vs alpha
def plotGridSearch(ax,scoresList):
    scoresList=copy(scoresList)
    scoresList.sort(key=lambda e:e["estimator"].getAlpha())

    ax.plot([ x["estimator"].getAlpha() for x in scoresList ],
            [ y["meanScore"] for y in scoresList ])

    ax.plot([ x["estimator"].getAlpha() for x in scoresList ],
            [ y["meanTrainScore"] for y in scoresList ])
            
    ax.set_xlabel("alpha")
    ax.set_ylabel("score")

    ax.legend(["test score","train score"])




