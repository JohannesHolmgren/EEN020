import numpy as np
import matplotlib.pyplot as plt

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
