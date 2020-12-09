#!/usr/bin/python
from math import log
import numpy as np
from scipy.spatial.distance import pdist, squareform


def compute_shape_context_1d(x, ranges):
    """ compute shape context of a time series of 1d data.
    Input:
        x: The time series of 1d data as a 2d numpy.array,
            where x[i, :] denotes its i-th data.
        ranges: The lower and upper range of shape context's bins
            as a tuple of size 2 float values.
            If not provided, range is simply (x.min(), x.max())
    Return:
        v: The computed shape context as a 2d numpy.array,
            where v[i, :] denotes the shape context of i-th data.
    """
    assert type(ranges) == list
    assert len(ranges) == 2

    assert len(x[0, :]) == 1

    # compute a distance matrix D,
    # where D[i, j] = log(dist(x[i, :], x[j, :]))
    D = squareform(pdist(x, 'euclidean') + np.finfo(np.float32).eps)
    np.fill_diagonal(D, np.finfo(np.float32).eps)
    D = np.log(D)

    # compute shapce contexts v,
    # where v[i, :] denotes the shape context of i-th data.
    num_data = len(x)
    bins = 10
    v = np.zeros((num_data, bins), dtype=int)
    for i in range(num_data):
        v[i, :] = np.histogram(D[i, :],
                               bins=bins,
                               range=ranges)[0]
    return v


def compute_shape_context_2d(x, ranges):
    """ compute shape context of a time series of 2d data.
    Input:
        x: The time series of 2d data as a 2d numpy.array,
            where x[i, :] denotes its i-th data.
        ranges: The lower and upper range of shape context's bins
            as a tuple of size 2 float values.
            If not provided, range is simply (x.min(), x.max())
    Return:
        v: The computed shape context as a 3d numpy.array,
            where v[i, :, :] denotes the shape context of i-th data.
    """
    # ranges must be a list of 2 float values
    assert type(ranges) == list
    assert len(ranges) == 2

    # x must be a set of 2d position defined as a 2d numpy.array
    assert len(x[0, :]) == 2

    # compute a distance matrix D,
    # where D[i, j] = log(dist(x[i, :], x[j, :]))
    D = squareform(pdist(x, metric='euclidean') + np.finfo(np.float32).eps)
    np.fill_diagonal(D, np.finfo(np.float32).eps)
    D = np.log(D)

    # compute a directinal angle matrix Theta.
    # where Theta[i, j] = angle(x[i, :], x[j, :])
    Theta = squareform(pdist(x, metric=lambda u, v: np.arctan2(v[1]-u[1], v[0]-u[0])))
    np.fill_diagonal(Theta, 0.0)

    # compute shapce contexts v,
    # where v[i, :] denotes the shape context of i-th data.
    num_data = len(x)
    bins = [10, 8]
    v = np.zeros((num_data, bins[0], bins[1]), dtype=int)
    for i in range(num_data):
        v[i, :, :] = np.histogram2d(D[i, :], Theta[i, :],
                                    bins=bins,
                                    range=ranges)[0]
    return v


def compute_shape_context_3d(x, ranges):
    """ compute shape context of a time series of 3d data.
    Input:
        x: The time series of 3d data as a 2d numpy.array,
            where x[i, :] denotes its i-th data.
        ranges: The lower and upper range of shape context's bins
            as a tuple of size 3 float values.
            If not provided, range is simply (x.min(), x.max())
    Return:
        v: The computed shape context as a 4d numpy.array,
            where v[i, :, :, :] denotes the shape context of i-th data.
    """
    # ranges must be a list of 3 float values
    assert type(ranges) == list
    assert len(ranges) == 3

    # x must be a set of 2d position defined as a 2d numpy.array
    assert len(x[0, :]) == 3

    # compute a distance matrix D,
    # where D[i, j] = log(dist(x[i, :], x[j, :]))
    D = squareform(pdist(x, metric='euclidean') + np.finfo(np.float32).eps)
    np.fill_diagonal(D, np.finfo(np.float32).eps)
    D = np.log(D)

    # compute a directinal angle matrix Theta.
    # where Theta[i, j] = angle_direction(x[i, :], x[j, :])
    Theta = squareform(pdist(x,
                             metric=lambda u, v:
                                 np.arctan2(v[1]-u[1], v[0]-u[0])))
    np.fill_diagonal(Theta, 0.0)

    # compute an elevation angle matrix Phi.
    # where Phi[i, j] = angle_elevation(x[i, :], x[j, :])
    Phi = squareform(pdist(x,
                           metric=lambda u, v:
                               np.arctan2(v[2]-u[2],
                                          np.sqrt((v[0]-u[0])**2 + (v[1]-u[1])**2))))
    np.fill_diagonal(Phi, 0.0)

    # compute shapce contexts v,
    # where v[i, :] denotes the shape context of i-th data.
    num_data = len(x)
    bins = [10, 8, 4]
    v = np.zeros((num_data, bins[0], bins[1], bins[2]), dtype=int)
    for i in range(num_data):
        x_polar = np.concatenate((D[i, :].reshape((-1, 1)),
                                  Theta[i, :].reshape((-1, 1)),
                                  Phi[i, :].reshape((-1, 1))),
                                 axis=1)
        v[i, :, :, :] = np.histogramdd(x_polar, bins=bins, range=ranges)[0]
    return v


def compute_shape_context(x, ranges):
    """ compute shape context of a time series of d-dimensional data.
    Input:
        x: The time series of 3d data as a 2d numpy.array,
            where x[i, :] denotes its i-th data.
        ranges: The lower and upper range of shape context's bins
            as a tuple of size d float values.
            If not provided, range is simply (x.min(), x.max())
    Return:
        v: The computed shape context as a d-dimensional numpy.array,
            where v[i, :, ..., :] denotes the shape context of i-th data.
    """
    dim = len(x[0, :])
    assert (dim > 0) and (dim < 4)

    # ranges must be a list of 3 float values
    assert type(ranges) == list
    if dim == 1:
        assert len(ranges) == 2
       # print(type(ranges[0]))
        assert (type(ranges[0]) == float) or (type(ranges[0]) == np.float64)
    else:
        assert len(ranges) == dim

    # x must be a set of 2d position defined as a 2d numpy.array
    assert len(x[0, :]) == dim

    # compute a distance matrix D,
    # where D[i, j] = log(dist(x[i, :], x[j, :]))
    D = squareform(pdist(x, metric='euclidean') + np.finfo(np.float32).eps)
    np.fill_diagonal(D, np.finfo(np.float32).eps)
    D = np.log(D)

    Theta = None
    Phi = None
    if dim > 1:
        # compute a directinal angle matrix Theta.
        # where Theta[i, j] = angle_direction(x[i, :], x[j, :])
        Theta = squareform(pdist(x, metric=lambda u, v: np.arctan2(v[1]-u[1], v[0]-u[0])))
        np.fill_diagonal(Theta, 0.0)
    if dim > 2:
        # compute an elevation angle matrix Phi.
        # where Phi[i, j] = angle_elevation(x[i, :], x[j, :])
        Phi = squareform(pdist(x, metric=lambda u, v: np.arctan2(v[2]-u[2], np.sqrt((v[0]-u[0])**2 + (v[1]-u[1])**2))))
        np.fill_diagonal(Phi, 0.0)

    # compute shapce contexts v,
    # where v[i, :] denotes the shape context of i-th data.
    num_data = len(x)
    bins = []
    shape = ()
    if dim == 1:
        bins = [10]
        shape = (num_data, bins[0])
    elif dim == 2:
        bins = [10, 8]
        shape = (num_data, bins[0], bins[1])
    elif dim == 3:
        bins = [10, 8, 4]
        shape = (num_data, bins[0], bins[1], bins[2])

    v = np.zeros(shape, dtype=int)
    for i in range(num_data):
        if dim == 1:
            v[i, :] = np.histogram(D[i, :], bins=bins[0], range=ranges)[0]
        if dim == 2:
            v[i, :, :] = np.histogram2d(D[i, :], Theta[i, :], bins=bins, range=ranges)[0]
        elif dim == 3:
            x_polar = np.concatenate((D[i, :].reshape((-1, 1)),
                                      Theta[i, :].reshape((-1, 1)),
                                      Phi[i, :].reshape((-1, 1))),
                                     axis=1)
            v[i, :, :, :] = np.histogramdd(x_polar, bins=bins, range=ranges)[0]
    return v


def find_ranges(x):

    log_distance = np.log(pdist(x, metric='euclidean') + np.finfo(np.float32).eps)
    log_distance_min = min(log_distance.min(), log(np.finfo(np.float32).eps))
    log_distance_max = log_distance.max()

    dim = len(x[0, :])
    if dim == 1:
        ranges = [log_distance_min, log_distance_max]
    else:
        # for min/max in distance
        ranges = [[log_distance_min, log_distance_max], 2*[0.0]]
        # for min/max in directional angle
        ranges[1][0] = -np.pi
        ranges[1][1] = np.pi
        if dim == 3:
            # for min/max in elevation angle
            ranges.append([-0.5 * np.pi, 0.5 * np.pi])

    return ranges
