###############
##Design the function "findRotMat" to  return 
# 1) rotMat1: a 2D numpy array which indicates the rotation matrix from xyz to XYZ 
# 2) rotMat2: a 2D numpy array which indicates the rotation matrix from XYZ to xyz 
# It is ok to add other functions if you need
###############

import numpy as np


# import cv2


def findRotMat(alpha, beta, gamma):
    def Rx(angle):
        return np.matrix([[1, 0, 0],
                          [0, np.cos(angle), -np.sin(angle)],
                          [0, np.sin(angle), np.cos(angle)]])

    def Ry(angle):
        return np.matrix([[np.cos(angle), 0, np.sin(angle)],
                          [0, 1, 0],
                          [-np.sin(angle), 0, np.cos(angle)]])

    def Rz(angle):
        return np.matrix([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]])

    a = np.radians(alpha)
    b = np.radians(beta)
    g = np.radians(gamma)

    Rf1 = Rz(g) @ Rx(b) @ Rz(a)
    rotMat1 = np.round((Rf1), decimals=3)

    # brute force
    # na = np.radians(360 - alpha)
    # nb = np.radians(360 - beta)
    # ng = np.radians(360 - gamma)
    # rf2 = Rz(na) @ Rx(nb) @ Rz(ng)
    # rotMat2 = np.round((rf2),decimals=3)

    rotMat2 = np.transpose(rotMat1)

    return rotMat1, rotMat2


if __name__ == "__main__":
    alpha = 45
    beta = 30
    gamma = 60
    rotMat1, rotMat2 = findRotMat(alpha, beta, gamma)
    print(rotMat1)
    print(rotMat2)
