import numpy as np
import math
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import cv2

def to_homogeneous(points: np.ndarray) -> np.ndarray:
    ''' Convert points to homogeneous coordinates by adding a 
    row of 1s to the coordinates. '''
    ones = np.ones(points.shape[1]).reshape(1, -1)
    return np.concat([points, ones])

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

def estimate_F_DLT(x1s: np.ndarray, x2s: np.ndarray) -> tuple:
    ''' Solve Homogeneous least squares for DLT equations using SVD.
        Use this result to find the fundamental camera F
    
    Parameters
    ----------
    x1s: np.ndarray
        The image points for camera 1

    x2s: np.ndarray
        The image points for camera 2

    Returns
    -------
    tuple:
        A tuple (F, M, v, sv_min) where F is the fundamental matrix,
        M is the M-matrix, v is the solution vector and sv_min is the 
        smallest singular value obtained from SVD.
    '''
    # Set up matrix M
    n_points = x1s.shape[1]
    dim = x1s.shape[0]
    M = np.zeros([n_points, dim * dim])
    for i in range(n_points):
        xx = np.outer(x2s[:,i], x1s[:,i])
        M[i,:] = xx.flatten()
    # Solve least squares problem
    U, S, Vh = np.linalg.svd(M)
    v = Vh[-1]
    sv_min = S[-1]
    F = v.reshape((3, 3))
    return (F, M, v, sv_min)

def enforce_fundamental(F_approx: np.ndarray) -> np.ndarray:
    ''' Enforce fundamental matrix to have zero determinant
    
    Does this by performing SVD and setting the minimum singular value
    to 0 before recreating F.

    Parameters
    ----------
    F_approx: np.ndarray
        The approximated fundamental matrix F
    
    Returns
    -------
    np.ndarray:
        The fundamental matrix F with zero determinant
    '''
    U, S, Vh = np.linalg.svd(F_approx)
    S[-1] = 0
    S_diag = np.diag(S)
    F = U @ S_diag @ Vh
    return F

def psphere(v):
    ''' Normalize vectors to lie on the unit sphere. '''
    norms = np.linalg.norm(v, axis=0)
    return v / norms

def compute_epipolar_errors(F: np.ndarray, x1s: np.ndarray, x2s: np.ndarray):
    ''' Compute distance between points and their corresponding epipolar lines 
    
    Parameters
    ----------
    F: np.ndarray
        The fundamental matrix

    x1s: np.ndarray
        Image points in image 1

    x2s: np.ndarray
        Image points in image 2

    '''
    # lines = F @ x1s
    lines = compute_epipolar_lines(F, x1s)
    distances = point_line_distance2D(x2s, lines)
    return distances

def get_epipolar_constraint(F: np.ndarray, x1s: np.ndarray, x2s: np.ndarray) -> float:
    ''' Check what the maximum error is for the epipolar constraint for all x1s and x2s. '''
    highest = 0
    for i in range(x1s.shape[1]):
        constraint = abs(x2s[:,i].T @ F @ x1s[:,i])
        if constraint > highest:
            highest = constraint
    return highest

def enforce_essential(E_approx: np.ndarray) -> np.ndarray:
    ''' Enforce the Essential matrix by setting its singular values to [1 1 0]. 
    
    '''
    U, S, Vh = np.linalg.svd(E_approx)
    S_diag = np.diag([1, 1, 0])
    return U @ S_diag @ Vh

def convert_E_to_F(E: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    ''' Get Fundamental matrix F from Essential matrix E.
    
    Using the formula K2^(T)FK1
    '''
    return K2.T @ E @ K1

def compute_epipolar_lines(F: np.ndarray, xs: np.ndarray) -> np.ndarray:
    ''' Comptue the epipolar lines for xs using 
    fundamental or essentail matrix F.

    Normalizes the epipolar lines to length 1 before returning them.
    
    Parameters
    ----------
    F: np.ndarray
       The fundamental or essential matrix for the two cameras
    xs: np.ndarray
        The image points in camera 1
    
    Returns
    -------
    np.ndarray:
        The epipolar lines in camera 2

    '''
    l = np.matmul(F, xs)
    l = pflat(l)
    l_norms = np.sqrt(l[0, :]**2 + l[1,:]**2)
    l = l / l_norms
    return l

def vl_sift(im: np.ndarray) -> tuple:
    ''' Perform SIFT on the image to find points of interest. 
    
    Parameters
    ----------
    im: np.ndarray
        The image to perform SIFT on

    Returns
    -------
    tuple[np.ndarray, np.ndarray]:
        the f matrix containing x, y, scale and orientation for each detected point
        and the SIFT descriptors
    '''
    # Initialize the SIFT detector
    sift = cv2.SIFT_create(contrastThreshold=0.01)  # 'PeakThresh' in VLFeat is similar to 'contrastThreshold' in OpenCV

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(im, None)

    # Convert keypoints to a format similar to VLFeat's `f1` matrix
    f = np.array([kp.pt + (kp.size, kp.angle) for kp in keypoints], dtype=np.float32).T  # Each column: [x, y, scale, orientation]
    d = descriptors.T  # SIFT descriptors
    return f, d

def vl_ubcmatch(d1, d2):
    ''' Find matches between two images' SIFT descriptors. '''
    # d1 and d2 are the SIFT descriptors from the two images
    # OpenCV FLANN-based matcher
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    # Match descriptors using k-nearest neighbors (k=2)
    matches = flann.knnMatch(d1.T, d2.T, k=2)  # Transpose descriptors to match OpenCV's format

    # Apply the Lowe's ratio test
    good_matches = []
    scores = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # Adjust the ratio as needed
            good_matches.append((m.queryIdx, m.trainIdx))
            scores.append(m.distance)

    # Convert matches and scores to NumPy arrays for easier manipulation
    matches = np.array(good_matches).T  # Shape: (2, num_matches)
    scores = np.array(scores)

    return matches, scores

def extract_P_from_E(E: np.ndarray) -> np.ndarray:
    ''' Get all camera matrix P from Essential matrix E. '''

    # Set upp all matrices needed
    U, _, Vh = np.linalg.svd(E)
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    u3 = U[:,2].reshape(-1, 1)

    # Assert det(U*V) > 0
    # Vh = Vh if np.linalg.det(U @ Vh) > 0 else -Vh
    if np.linalg.det(Vh) < 0:
        Vh = -Vh

    # Compute the four camera solutions
    P21 = np.hstack([U @ W   @ Vh,  u3])
    P22 = np.hstack([U @ W   @ Vh, -u3])
    P23 = np.hstack([U @ W.T @ Vh,  u3])
    P24 = np.hstack([U @ W.T @ Vh, -u3])
    return np.array([P21, P22, P23, P24])




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
