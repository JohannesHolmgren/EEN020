import numpy as np
import matplotlib.pyplot as plt

import utils

# Function to plot line in 2D
def plot_line_2D(ax: plt.axes, p1:np.ndarray, p2:np.ndarray) -> plt.axes:

    ''' Plot a line onto ax between p1 and p2
    Parameters
    ----------
    ax: plt.axes
        The matplotlib surface to plot on (from fig, ax = plt.subplots())

    p1: np.ndarray
        Start point in normalized homogeneous coordinates

    p2: np.ndarray
        End point in normalized homogeneous coordinates
    
    Returns
    -------
    plt.axes
        The surface plotted on. Can be used for further plotting
    '''

    ax.plot((p1[0], p2[0]), (p1[1], p2[1]))
    return ax

# Function to plot line in 2D
def plot_line_3D(ax: plt.axes, p1:np.ndarray, p2:np.ndarray) -> plt.axes:
    ''' Plot a line onto ax between p1 and p2

    Parameters
    ----------
    ax: plt.axes
        The matplotlib surface to plot on (from fig, ax = plt.subplots())

    p1: np.ndarray
        Start point in normalized homogeneous coordinates

    p2: np.ndarray
        End point in normalized homogeneous coordinates
    
    Returns
    -------
    plt.axes
        The surface plotted on. Can be used for further plotting
    '''

    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])
    return ax

# Function to plot coordinates in 2D
def plot_points_2D(ax: plt.axes, points: np.ndarray, c=None, s=1, rev=False) -> None:
    ''' Plot points in 2D space

    Plots all 'n' points (2 x n) using matplotlib.

    Parameters
    ----------
    ax: plt.axes
        The matplotlib surface to plot on (from fig, ax = plt.subplots())

    points: np.ndarray
        A 2 x n matrix representing 'n' points

    c: str (default None)
        A color string to pass to matplotlib. If None then use y-value.

    s: float (default 1)
        Scale of the points

    Returns
    plt.axes
    -------
        The surface plotted on. Can be used for further plotting

    ''' 
    if rev:
        ax.scatter(points[1,:], points[0,:], c=c, s=s)
    else:
        ax.scatter(points[0,:], points[1,:], c=c, s=s)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return ax

# Function to plot coordinates in 3D
def plot_points_3D(ax: plt.axes, points: np.ndarray, c=None, s=1) -> None:
    ''' Plot points in 3D space

    Plots all 'n' points (3 x n) using matplotlib.

    Parameters
    ----------
    ax: plt.axes
        The matplotlib surface to plot on (from fig, ax = plt.subplots())
    
    points: np.ndarray
        A 3 x n matrix representing 'n' points

    c: str (default None)
        A color string to pass to matplotlib. If None then use y-value.

    s: float (default 1)
        Scale of the points
    
    Returns
    -------
    plt.axes
        The surface plotted on. Can be used for further plotting
    
    '''
    ax.scatter(points[0,:], points[1,:], points[2,:], c=c, s=s)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_camera(ax: plt.axes, center: np.ndarray, axis: np.ndarray, scale: float) -> None:
    ''' Plots the principal axis of a camera scaled by s starting from the camera center

    Camera center and axis can be obtained from the function 'camera_center_and_axis'.

    Parameters
    ----------
    ax: plt.axes
        The matplotlib surface to plot on (from fig, ax = plt.subplots())

    center: np.ndarray
        The camera center

    axis: np.ndarray
        The camera's principal axis
    
    scale: float
        The length (scale) of the camera principal axis    
    
    '''
    ax.quiver(center[0], center[1], center[2], 
              axis[0], axis[1], axis[2], 
              length=scale, arrow_length_ratio=0.3, color='r'
    )

def plot_cams(ax: plt.axes, centers: np.ndarray, axes: np.ndarray, scale: float) -> None:
    ''' Plot all cameras using their principal axis and their centers. 
    
    Camera center and axis can be obtained from the function 'camera_center_and_axis'.

    Parameters
    ----------
    ax: plt.axes
        The matplotlib surface to plot on (from fig, ax = plt.subplots())

    centers: np.ndarray
        The camera centers

    axes: np.ndarray
        The camera's principal axes
    
    scale: float
        The length (scale) of the camera principal axes

    '''
    for center, axis in zip(centers, axes):
        plot_camera(ax, center, axis, scale)

def rital(ax: plt.axes, lines: np.ndarray):
    ''' Draw lines in an image
    
    Parameters
    ----------
    ax: plt.axes
        The matplotlib.pyplot axis to plot in

    lines: np.ndarray
        The homogeneous lines to plot
    
    '''
    if lines.size == 0:
        return
    n_lines = lines.shape[1]
    # Compute direction vectors orthogonal to the lines
    directions = utils.psphere(np.array([lines[1,:], -lines[0,:], np.zeros(n_lines)]))
    # Compute points on the lines by finding intersection with z=0
    points = utils.pflat(np.cross(directions.T, lines.T).T)
    for i in range(n_lines):
        x_vals = [points[0, i] - 2000 * directions[0, i], points[0, i] + 2000 * directions[0, i]]
        y_vals = [points[1, i] - 2000 * directions[1, i], points[1, i] + 2000 * directions[1, i]]
        ax.plot(x_vals, y_vals)


def plot_points_and_lines(ax: plt.axes, xs: np.ndarray, ep_lines: np.ndarray, im: np.ndarray) -> None:
    ''' Plot image points and their corresponding epipolar lines
        together with the image. 
        
    Parameters
    ----------
    ax: plt.axes
        The matplotlib surface to plot on (from fig, ax = plt.subplots())

    xs: np.ndarray:
        The image points to plot
    
    ep_lines: np.ndarray:
        The epipolar lines corresponding to the image points xs
    
    im: np.ndarray
        The image to be plotted, as a numpy array

    '''
    # Set window
    ax.set_aspect('equal')
    ax.set_xlim(0, im.shape[1])
    ax.set_ylim(im.shape[0], 0)
    # Plot image
    ax.imshow(im)
    # Plot points
    plot_points_2D(ax, xs, c='r', s=10)
    # Plot epipolar lines
    rital(ax, ep_lines)

if __name__ == '__main__':
    fig, ax = plt.subplots()
    points = np.array([[0, 1], [0, 2]])
    points_3d = np.array([[0, 1], [0, 1], [0, 1]])
    plot_points_2D(ax, points)
    plot_line_2D(ax, points[:,0], points[:,1])

    fig = plt.figure()
    ax_3D = fig.add_subplot(projection='3d')
    plot_points_3D(ax_3D, points_3d)
    plot_line_3D(ax_3D, points_3d[:,0], points_3d[:,1])

    plt.show()
