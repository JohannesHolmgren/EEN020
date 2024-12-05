import numpy as np
import math
from scipy.linalg import null_space

# Computer Exercise 1
def pflat(matrix: np.ndarray) -> np.ndarray:
    ''' Normalize 'n' points of dimension 'm' 

    Normalized by dividing by the last item for each point.
    It is assumed that the last coordinate is non-zero. 

    Parameters
    ----------
    matrix: np.ndarray
        A m x n matrix 
    
    Returns
    -------
    np.ndarray
        A normalized matrix of size m x n where the last element is 1
    '''
    return matrix / matrix[-1]


def point_line_distance2D(points: np.ndarray, lines: np.ndarray) -> np.ndarray:
    ''' Return the distance between a line and a point in 2D cartesian coordinates.

    Parameters
    ----------
    points: np.ndarray
        An array of (normalized) homogeneous points in P2

    lines: np.ndarray
        An array of lines on the shape (a, b, c)

    Returns
    -------
        The distance between the point and line

    '''
    nominators = np.abs(np.sum(points * lines, axis=0))
    denominators = np.sqrt(lines[0,:]**2 + lines[1,:]**2)
    return nominators / denominators

# Computer Exercise 3
def camera_center_and_axis(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ''' Get camera center and principle axis of the camera matrix 'P'

    Parameters
    ----------
    P: np.ndarray
        The camera matrix
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray] 
        The first array is the camera center in cartesian coordinates
        The second array is the principal axis normalized to length 1
        
    '''
    # The camera center is the null space of the camera matrix
    camera_center = null_space(P)
    # The principal axis is the third row of R
    camera_center_cart = (camera_center / camera_center[-1])[0:-1]
    principal_axis = P[2,0:3]
    principal_axis_norm = principal_axis / np.linalg.norm(principal_axis)

    return (camera_center_cart, principal_axis_norm)


def flip_points(xs: np.ndarray, height: np.ndarray) -> np.ndarray:
    ''' Flip points along their y-axis

    Parameters
    ----------
    xs: np.ndarray
        The points to flip

    height : float
        The height of the image

    Returns
    -------
    np.ndarray
        The flipped points
    
    '''
    flipped = xs.copy()
    flipped[1, :] = height - xs[1,:]
    return flipped


def get_normalization_matrix(xs: np.ndarray) -> np.ndarray:
    ''' Get the normalization matrix which subtracts mean and divides by std_dev. 

    Takes the mean and std_dev for x-values and y-values separately.

    Parameters
    ----------
    xs: np.ndarray
        The points to normalize
    
    Returns
    -------
    np.ndarray
        The normalization matrix

    '''
    meanX = np.mean(xs[0])
    meanY = np.mean(xs[1])
    stdX  = np.std(xs[0])
    stdY  = np.std(xs[1])

    N = np.array([
        [1 / stdX,        0, -meanX / stdX],
        [0,        1 / stdY, -meanY / stdY],
        [0,               0,             1]
    ])
    return N


def triangulate_3D_point_DLT(xs1: np.ndarray, xs2: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    ''' Triangulate 3D points X projected into x1 and x2 using P1 and P2.

    Parameters
    ----------
    xs1: np.ndarray
        Points projected into P1 in homogeneous coordinates (3 x N)

    xs2: np.ndarray
        Points projected into P2 in homogeneous coordinates (3 x N)

    P1: np.ndarray
        Camera matrix for camera 1 (3 x 4)
    
    P2: np.ndarray
        Camera matrix for camera 2 (3 x 4)

    Returns
    -------
    np.ndarray
        The triangulated 3D points in homogeneous coordinates (4 x N)
    '''
    Xs = []
    for x1, x2 in zip(xs1.T, xs2.T):

        zeros = np.zeros_like(x1).reshape(-1, 1)
        
        M = np.vstack([
            np.hstack([P1, -x1.reshape(-1, 1), zeros]),
            np.hstack([P2, zeros, -x2.reshape(-1, 1)])
        ])
        U, S, Vh = np.linalg.svd(M)
        v = Vh[-1]
        Xs.append(v[:4])
    Xs = np.array(Xs).T
    Xs = pflat(Xs)
    return Xs

    Xs = []
    for x1, x2 in zip(xs1.T, xs2.T):
        A = np.array([
            x1[0] * P1[2, :] - P1[0, :],
            x1[1] * P1[2, :] - P1[1, :],
            x2[0] * P2[2, :] - P2[0, :],
            x2[1] * P2[2, :] - P2[1, :]
        ])
        
        # Solve the system AX = 0 using SVD
        _, _, Vh = np.linalg.svd(A)
        X = Vh[-1]  # The last row of Vh corresponds to the solution
        Xs.append(X / X[-1])  # Normalize to make homogeneous

    return np.array(Xs).T  # Return as a 4 x N array


if __name__ == '__main__':
    P1 = np.eye(3, 4)
    P2 = np.array([
        [ 0.62208686  ,0.27002211  ,0.73491224  ,0.92229092],
        [ 0.28035671  ,-0.95323078 ,0.11292118  ,0.14220198],
        [ 0.73103218  ,0.13579079  ,-0.66869485  ,0.35938565]
    ])
    x1 = np.array([145.56 ,466.02 ,1])
    x2 = np.array([81.98 ,484.7 ,1])


    # print(np.hstack([x1.reshape(-1, 1), np.zeros_like(x1).reshape(-1, 1)]))
    
    triangulate_3D_point_DLT(np.array(np.array([x1]).reshape(-1, 1)), np.array(np.array([x2]).reshape(-1, 1)), P1, P2)
