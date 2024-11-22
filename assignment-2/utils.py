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


def point_line_distance2D(point: np.ndarray, line: np.ndarray) -> np.ndarray:
    ''' Return the distance between a line and a point in 2D cartesian coordinates.

    Parameters
    ----------
    point: np.ndarray
        A 2D point in R2 or P2 (must be normalized for P2)

    line: np.ndarray
        A line on the shape (a, b, c)

    Returns
    -------
        The distance between the point and line

    '''
    a, b, c = line
    x1, x2 = point[0:2]
    return abs(a*x1 + b*x2 + c) / math.sqrt(a**2 + b**2)

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