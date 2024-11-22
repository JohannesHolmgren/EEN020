# Import external modules
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.linalg import null_space, inv
from cv2 import imread

# Import internal modules/files
import plotter


# Load all matrix data
MAT_PATH = 'data/mat/'

compEx1_data = loadmat(MAT_PATH + 'compEx1.mat')
compEx2_data = loadmat(MAT_PATH + 'compEx2.mat')
compEx3_data = loadmat(MAT_PATH + 'compEx3.mat')
compEx4_data = loadmat(MAT_PATH + 'compEx4.mat')

x2D = compEx1_data['x2D']
x3D = compEx1_data['x3D']

p1 = compEx2_data['p1']
p2 = compEx2_data['p2']
p3 = compEx2_data['p3']

U = compEx3_data['U']
P1 = compEx3_data['P1'] 
P2 = compEx3_data['P2'] 

# Extract variables for computer exercise 4
K = compEx4_data['K']
v = compEx4_data['v']
corners = compEx4_data['corners']

# Load all images
IMG_PATH = 'data/images/'

compEx2 = imread(IMG_PATH + 'compEx2.jpg')
compEx3im1 = imread(IMG_PATH + 'compEx3im1.jpg')
compEx3im2 = imread(IMG_PATH + 'compEx3im2.jpg')
compEx4 = imread(IMG_PATH + 'compEx4.jpg')


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
    

def plot_camera(ax: plt.axes, P: np.ndarray, scale: float) -> None:
    ''' Plots the principal axis of a camera scaled by s

    Internally calls on 'camera_center_and_matrix' to get the
    center and principal axis of the camera.

    Parameters
    ----------
    ax: plt.axes
        The matplotlib surface to plot on (from fig, ax = plt.subplots())

    P: np.ndarray
        The camera matrix
    
    scale: float
        The length (scale) of the camera principal axis    
    
    '''
    center, axis = camera_center_and_axis(P)
    axis = axis
    ax.quiver(center[0], center[1], center[2], axis[0], axis[1], axis[2], length=scale, arrow_length_ratio=0.3, color='r')

# Computer Exercise 4
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.tight_layout()
ax1.set_aspect('equal')

# Show image
ax1.imshow(compEx4)
# Plot points
plotter.plot_points_2D(ax1, corners, c='r')

### Answer to question: The origin is in the upper left corner

# Calibrate camera
K_inv = inv(K)
corners_norm = np.matmul(K_inv, corners)

plotter.plot_points_2D(ax2, corners_norm, c='r')
ax2.invert_yaxis()

### Answer to question: The origin of the image coordinate system is 
### now located slightly to the left of the middle of the image.

# The corners in 3D is given by finding s so that U = (x s) is orthogonal to plane
v_norm = pflat(v)
corners_3D = np.array([c / -np.matmul(v_norm.T[:,:-1], c) for c in corners_norm.T]).T
corners_3D_hom = np.vstack([corners_3D, np.ones((1, corners_3D.shape[1]))])

# Get camera center and principal axis. P is now [I 0] after calibration
P = np.array(np.hstack([np.identity(3), np.zeros((3, 1))]))
center, axis = camera_center_and_axis(P)

# Plot Camera and 3D corners
fig_3D = plt.figure()
ax_3D = fig_3D.add_subplot(111, projection='3d')
plot_camera(ax_3D, P, 1)
plotter.plot_points_3D(ax_3D, corners_3D_hom)
ax_3D.set_aspect('equal')

### Answer to question: Yes, the 3D points look reasonable compared to their 2D projection.

# Compute new Camera P2
R = np.array([
    [np.cos(np.pi/5), 0, -np.sin(np.pi/5)],
    [0              , 1,                0],
    [np.sin(np.pi/5), 0,  np.cos(np.pi/5)]
])
C_P2 = np.array([-2.5, 0, 0])
# t is given by solving 0 = RC + t <=> t = -RC
t = -np.matmul(R, C_P2).reshape(3, 1)
P2 = np.hstack([R, t])

# Plot new camera
plot_camera(ax_3D, P2, 1)

# Transform the normalized corner points to using the homography H = (R - t*v_norm^T)
H = (R - np.matmul(t, v_norm[:-1].T))
corners_new = np.array([np.matmul(H, c) for c in corners_norm.T]).T
corners_new_norm = pflat(corners_new)

# Plot transformed points
fig, ax = plt.subplots()
ax.invert_yaxis()
plotter.plot_points_2D(ax, corners_new_norm, c='r')

# Project 3D points onto P2
corners_proj_P2 = np.array([np.matmul(P2, p) for p in corners_3D_hom]).T
corners_proj_P2_norm = corners_proj_P2 / corners_proj_P2[2,:]

# Plot projected points in P2
plotter.plot_points_2D(ax, corners_proj_P2_norm[:2,:])

print(corners_3D_hom)
print()
print(corners_proj_P2)
print()
print(corners_proj_P2_norm[:2,:])
print()
print(corners_new_norm)

### Show everything
plt.show()
