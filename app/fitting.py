# the following code is taken from https://github.com/brikeats/Snakes-in-a-Plane

import cv2
import numpy as np
from scipy.optimize import minimize


def snake_energy(flattened_pts, edge_dist, alpha, beta):
    pts = np.reshape(flattened_pts, (int(len(flattened_pts)/2), 2))

    # external energy (favors low values of distance image)
    dist_vals = ndimage.interpolation.map_coordinates(edge_dist, [pts[:,0], pts[:,1]], order=1)
    edge_energy = np.sum(dist_vals)
    external_energy = edge_energy

    # spacing energy (favors equi-distant points)
    prev_pts = np.roll(pts, 1, axis=0)
    next_pts = np.roll(pts, -1, axis=0)
    displacements = pts - prev_pts
    point_distances = np.sqrt(displacements[:,0]**2 + displacements[:,1]**2)
    mean_dist = np.mean(point_distances)
    spacing_energy = np.sum((point_distances - mean_dist)**2)

    # curvature energy (favors smooth curves)
    curvature_1d = prev_pts - 2*pts + next_pts
    curvature = (curvature_1d[:,0]**2 + curvature_1d[:,1]**2)
    curvature_energy = np.sum(curvature)

    return external_energy + alpha*spacing_energy + beta*curvature_energy


def snake(pts, edge_dist, alpha=0.5, beta=0.25, nits=100, point_plot=None):
    if point_plot:
        def callback_function(new_pts):
            callback_function.nits += 1
            y = new_pts[0::2]
            x = new_pts[1::2]
            point_plot.set_data(x,y)
            plt.title('%i iterations' % callback_function.nits)
            point_plot.figure.canvas.draw()
            plt.pause(0.02)
        callback_function.nits = 0
    else:
        callback_function = None

    # optimize
    cost_function = partial(snake_energy, alpha=alpha, beta=beta, edge_dist=edge_dist)
    options = {'disp':False}
    options['maxiter'] = nits  # FIXME: check convergence
    method = 'BFGS'  # 'BFGS', 'CG', or 'Powell'. 'Nelder-Mead' has very slow convergence
    res = optimize.minimize(cost_function, pts.ravel(), method=method, options=options, callback=callback_function)
    optimal_pts = np.reshape(res.x, (int(len(res.x)/2), 2))

    return optimal_pts
