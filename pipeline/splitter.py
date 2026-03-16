import numpy as np


def split_exposure(image: np.ndarray):
    """
    线阵相机高低曝光拆分
    偶数行：高曝光
    奇数行：低曝光
    """

    low = image[0::2]
    high = image[1::2]

    return high, low