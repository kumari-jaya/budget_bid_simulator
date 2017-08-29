import numpy as np



def divideFloat(numerator, denominator):
    res = numerator / denominator
    if np.isnan(res):
        return 0.0
    if np.isinf(res):
        return 0.0
    return res

def dividePotentialClicks(numerator,denominator):
    div = numerator/denominator
    div[np.isnan(div)]=0
    div[np.isinf(div)]=0
    return div

