###############
##Design the function "calibrate" to  return 
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates. 
#                                            True if the intrinsic parameters are invariable.
# It is ok to add other functions if you need
###############
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners


def calibrate(imgname):
    img = imread(imgname)
    gray = cvtColor(img, COLOR_BGR2GRAY)

    criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obpoints = []
    imgpoints = []

    w = np.array([[40.0, 0.0, 40.0], [40.0, 0.0, 30.0], [40.0, 0.0, 20.0], [40.0, 0.0, 10.0],
                  [30.0, 0.0, 40.0], [30.0, 0.0, 30.0], [30.0, 0.0, 20.0], [30.0, 0.0, 10.0],
                  [20.0, 0.0, 40.0], [20.0, 0.0, 30.0], [20.0, 0.0, 20.0], [20.0, 0.0, 10.0],
                  [10.0, 0.0, 40.0], [10.0, 0.0, 30.0], [10.0, 0.0, 20.0], [10.0, 0.0, 10.0],
                  [0.0, 0.0, 40.0], [0.0, 0.0, 30.0], [0.0, 0.0, 20.0], [0.0, 0.0, 10.0],
                  [0.0, 10.0, 40.0], [0.0, 10.0, 30.0], [0.0, 10.0, 20.0], [0.0, 10.0, 10.0],
                  [0.0, 20.0, 40.0], [0.0, 20.0, 30.0], [0.0, 20.0, 20.0], [0.0, 20.0, 10.0],
                  [0.0, 30.0, 40.0], [0.0, 30.0, 30.0], [0.0, 30.0, 20.0], [0.0, 30.0, 10.0],
                  [0.0, 40.0, 40.0], [0.0, 40.0, 30.0], [0.0, 40.0, 20.0], [0.0, 40.0, 10.0]])

    ret, corners = findChessboardCorners(gray, (4, 9), None)

    # w += 60
    
    if ret:
        corners = corners.reshape(-1, 2)
        imgpoints.append(corners)

        obpoints.append(w)

        corners2 = cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        drawChessboardCorners(img, (4, 9), corners2, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    xyz = np.array(obpoints[0])
    uv = np.array(imgpoints[0])

    N = len(xyz)
    M = np.zeros((2 * N, 12), dtype=np.float32)

    for i in range(N):
        X, Y, Z = xyz[i]
        u, v = uv[i]

        row1 = np.array([X, Y, Z, 1, 0, 0, 0, 0, -X * u, -Y * u, -Z * u, -u])
        row2 = np.array([0, 0, 0, 0, X, Y, Z, 1, -X * v, -Y * v, -Z * v, -v])

        M[2 * i] = row1
        M[(2 * i) + 1] = row2

    u, s, vt = np.linalg.svd(M)

    mval = vt[11]
    lam = 1/(np.linalg.norm(mval[8:-1]))

    m = lam * mval
    m = m.reshape(3, 4)

    m1 = m[0][0:-1]
    m2 = m[1][0:-1]
    m3 = m[2][0:-1]

    ox = (m1.T @ m3)
    oy = (m2.T @ m3)
    fx = (np.sqrt((m1.T @ m1) - (ox ** 2)))
    fy = (np.sqrt((m2.T @ m2) - (oy ** 2)))

    in_params = [fx, fy, ox, oy]
    return in_params, True


if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')
    print(intrinsic_params)
    print(is_constant)
