import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    # Hint 1 - Determine nominal time for each point in the path using V_des
    # path_length = np.sum(np.diff(path,axis=0),axis=0)
    # nominal_time = 
    path = np.array(path)
    differences = np.linalg.norm(np.diff(path,axis=0),axis=1)
    differences = np.concatenate(([0],differences))
    accumulated_length = np.cumsum(differences)
    # total_time = accumulated_length[-1]/V_des
    nominal_time = accumulated_length/V_des
    # Hint 2 - Use splrep to determine cubic coefficients that best fit given path in x, y
    # print(path[1:4,0])
    # print(nominal_time[1:4])
    x_curve_coeffs = scipy.interpolate.splrep(nominal_time, path[:,0], s=alpha, k=3)
    y_curve_coeffs = scipy.interpolate.splrep(nominal_time, path[:,1], s=alpha, k=3)
    # Hint 3 - Use splev to determine smoothed paths. The "der" argument may be useful.
    t_smoothed = np.arange(0,nominal_time[-1], dt)
    x_d = scipy.interpolate.splev(t_smoothed, x_curve_coeffs, der=0)
    y_d = scipy.interpolate.splev(t_smoothed, y_curve_coeffs, der=0)
    xd_d = scipy.interpolate.splev(t_smoothed, x_curve_coeffs, der=1)
    yd_d = scipy.interpolate.splev(t_smoothed, y_curve_coeffs, der=1)
    theta_d = np.arctan2(yd_d,xd_d)
    xdd_d = scipy.interpolate.splev(t_smoothed, x_curve_coeffs, der=2)
    ydd_d = scipy.interpolate.splev(t_smoothed, y_curve_coeffs, der=2)
    
    # t_smoothed = nominal_time
    ########## Code ends here ##########
    traj_smoothed = np.stack([x_d, y_d, theta_d, xd_d, yd_d, xdd_d, ydd_d]).transpose()

    return traj_smoothed, t_smoothed