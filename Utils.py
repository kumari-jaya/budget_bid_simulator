

import numpy as np


def divideFloat(numerator, denominator):
    """

    :param numerator:
    :param denominator:
    :return:
    """
    div = numerator / denominator
    div[np.isnan(div)] = 0
    div[np.isinf(div)] = 0

    return div