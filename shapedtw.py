#!/usr/bin/python
import numpy as np

from fastdtw import fastdtw

from shape_context import find_ranges, compute_shape_context


def matching_fastdtw(x, y, dist):
    """ matches a two time series of dim-dimensional data x and y.
        Input:
            x: A time series of a dim-dimensional data as a 2d numpy.array,
                where x[i, :] denotes its i-th data.
            y: A time series of a dim-dimensional data as a 2d numpy.array,
                where y[i, :] denotes its i-th data.
        Return:
            dist: The total distance between the sequentes x and y.
            correspondences: A set of established correspondences,
            where x[correspondences[i][0], :] and y[correspondences[i][1]] are
            i-th corresponding points.
    """
    assert x.ndim == y.ndim
    assert len(x[0, :]) == len(y[0, :])

    dist, correspondences = fastdtw(x, y, dist=dist)
    return dist, correspondences


def matching_shapedtw(x, y, dist):
    """ matches a two time series of dim-dimensional data x and y by FastDTW.
        Input:
            x: A time series of a dim-dimensional data as a 2d numpy.array,
                where x[i, :] denotes its i-th data.
            y: A time series of a dim-dimensional data as a 2d numpy.array,
                where y[i, :] denotes its i-th data.
        Return:
            dist: The total distance between the sequentes x and y.
            correspondences: A set of established correspondences,
            where x[correspondences[i][0], :] and y[correspondences[i][1]] are
            i-th corresponding points.
    """
    #assert x.ndim == y.ndim
    #assert len(x[0, :]) == len(y[0, :])

    # compute shape context
    ranges = find_ranges(np.concatenate((x, y)))
    feature_x = compute_shape_context(x, ranges)
    feature_y = compute_shape_context(y, ranges)

    # reshape the feature tensor of arbitrary order to matrix
    # because fastdtw only accepts 2d numpy.arrays.
    return fastdtw(feature_x.reshape(len(feature_x), -1),
                   feature_y.reshape(len(feature_y), -1),
                   dist=dist)


def matching_1d(x, y, dist, bins=[10]):
    assert x.ndim == y.ndim

    return fastdtw(x, y, dist=dist)


def matching_2d(x, y, dist, bins=[10, 8]):
    assert x.ndim == y.ndim

    return fastdtw(x, y, dist=dist)


def matching_3d(x, y, dist, bins=[10, 8, 4]):
    assert x.ndim == y.ndim

    dist, correspondences = fastdtw(x, y, dist=dist)
    return dist, correspondences
