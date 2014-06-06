""" This file contains several functions useful for plotting figures using
matplotlib """

from __future__ import division

import itertools

import numpy as np
from scipy import interpolate, ndimage

import matplotlib.pyplot as plt
import matplotlib.colors as mclr
#import matplotlib.path as mpath


COS45 = np.cos(0.25*np.pi)


def _get_regions(values):
    """ returns an array of ids for each distinct region """
    weights = 2**np.arange(0, values.shape[2])
    return np.dot(values, weights)


def scale_image(img_in, dim_out):
    """
    Scales an image to the new_dimensions `dim_out`.
    Taken from http://stackoverflow.com/q/5586719/932593
    """

    # process input
    img_in = np.atleast_2d(img_in)
    dim_in = img_in.shape

    # setup interpolation object
    x = np.arange(0, dim_in[0])
    y = np.arange(0, dim_in[1])
    interp = interpolate.RectBivariateSpline(x, y, img_in, kx=2, ky=2)

    # calculate the new image
    xx = np.linspace(0, x.max(), dim_out[0])
    yy = np.linspace(0, y.max(), dim_out[1])
    return interp(xx, yy)


def get_contour_plot_hatched(values, colors, **kwargs):
    """
    Takes three dimensional boolean data and projects out the last dimension,
    by using hatched regions to symbolize parts, where several variables are
    true. The function returns an array with colors at each point.
    The style of the image can be influenced by these parameters:

    background_color    Color chosen, if all entries are False
    stripe_width        Width of the stripes indicating overlapping regions
    stripe_orientation  Orientation of the stripes ('\', '-', '/', or '|')

    boundary            True, if the boundary should be plotted
    boundary_width      Width of the boundary in pixel
    """

    # process general parameters
    background_color = kwargs.pop('background_color', 'w')
    nan_color = kwargs.pop('nan_color', 'k')
    return_all = kwargs.pop('return_all', False)

    # process parameters of the stripes
    stripe_width = kwargs.pop('stripe_width', 0.05)
    if stripe_width < 1:
        stripe_width = int(stripe_width*max(values.shape[:2]))
    stripe_width_diag = stripe_width/COS45
    stripe_orientation = kwargs.pop('stripe_orientation', '/')

    # process parameters of the boundary
    boundary = kwargs.pop('boundary', False)
    boundary_width = kwargs.pop('boundary_width', 1)

    # build orientation list to unify input
    dimensions = len(values[0, 0, :])
    if hasattr(stripe_orientation, '__iter__'):
        orientations = itertools.cycle(stripe_orientation)
        orientations = list(itertools.islice(orientations, dimensions))
    else:
        orientations = [stripe_orientation]*dimensions

    # convert the input and calculate the number of input values
    values = np.atleast_3d(values)
    nums = values.sum(2).astype(int)
    nan_as_int = np.array(np.nan).astype(int)

    # translate the color to RGB values
    get_color = mclr.ColorConverter().to_rgb
    background_color = get_color(background_color)
    nan_color = get_color(nan_color)
    colors = [get_color(c) for c in colors]

    # check, if the boundary values have to be calculated
    if boundary:
        # associate each region with a unique float number
        regions = _get_regions(values)
        regions[np.isnan(regions)] = -1

        # construct a filter for boundary detection
        w = np.ceil(boundary_width)
        coords = np.arange(-2*w, 2*w+1)
        x, y = np.meshgrid(coords, coords)
        r = np.sqrt(x**2 + y**2)
        sigma = 0.5*boundary_width + 0.5
        filter_matrix = 0.5 - 0.5*np.tanh(2*(r - sigma))

        # convolute the image with the filter for each region to get boundaries
        boundary_point = np.zeros(values.shape[:2])
        for k in np.unique(regions):
            panels = (regions != k)
            edges = ndimage.convolve(panels.astype(float), filter_matrix)
            boundary_point[~panels] += edges[~panels]

        boundary_point[boundary_point > 1] = 1.

    else:
        # no boundary points requested
        boundary_point = np.zeros(values.shape[:2])

    # define the different orientations of the stripe pattern
    orientation_functions = {
        '\\': lambda x, y: ((x - y) % stripe_width_diag)*nums[x, y]/stripe_width_diag,
        '-':  lambda x, y: (x % stripe_width)*nums[x, y]/stripe_width,
        '/':  lambda x, y: ((x + y) % stripe_width_diag)*nums[x, y]/stripe_width_diag,
        '|':  lambda x, y: (y % stripe_width)*nums[x, y]/stripe_width
    }

    # iterate over all pixels and process them individually
    res = np.zeros((values.shape[0], values.shape[1], 3))
    for x, y in np.ndindex(values.shape[:2]):
        if nums[x, y] == nan_as_int:
            res[x, y, :] = nan_color
        elif nums[x, y] == 0:
            res[x, y, :] = background_color
        else:
            # choose a color index based on the current stripe
            try:
                i = orientation_functions[orientations[nums[x, y] - 1]](x, y)
            except KeyError:
                raise ValueError(
                    'Allowed stripe orientation values: %s'
                        %(', '.join(orientation_functions.keys()))
                )

            # choose the color from the current values
            color_index = np.nonzero(values[x, y, :])[0][int(i)]
            res[x, y] = colors[color_index]

    # change color, if it is a boundary point
    res *= (1 - boundary_point[:, :, np.newaxis])

    if return_all:
        return res, kwargs
    else:
        return res


def make_contour_plot_hatched(values, colors, **kwargs):
    """
    Takes three dimensional boolean data and projects out the last dimension,
    by using hatched regions to symbolize parts, where several variables are
    true. The function uses imshow to plot the image to the current axes.
    The style of the image can be influenced by these parameters:

    background_color    Color chosen, if all entries are False
    stripe_width        Width of the stripes used in overlapping regions
    stripe_orientation  Orientation of the stripes ('\', '-', '/', or '|')

    boundary            True, if the boundary should be plotted
    boundary_color      Color of the boundary
    boundary_style      Dictionary of additional styles used in the contour
                        plot of the boundary
    boundary_smoothing  Float number determining the smoothing of the boundary

    """
    # read parameters
    boundary = kwargs.pop('boundary', True)
#     boundary_color = kwargs.pop('boundary_color', 'k')
    boundary_style = kwargs.pop('boundary_style', {})
    boundary_smoothing = kwargs.pop('boundary_smoothing', 1)

    # extract parameters used in both the imshow and the contour plot
    plot_parameters = {
        val: kwargs.pop(val, None)
        for val in ('alpha', 'origin', 'extent')
    }
    boundary_style.update(plot_parameters)

    # calculate the plot
    img, imshow_kwargs = get_contour_plot_hatched(
        values, colors, return_all=True, **kwargs
    )
    # do the plot of the image using
    imshow_kwargs.update(plot_parameters)
    plt.imshow(img, **imshow_kwargs)

    # plot boundaries if requested
    if boundary:
        regions = _get_regions(values)
        # plot a contour plot for each region
        for k in np.unique(regions):
            data = (regions == k).astype(float)
            if boundary_smoothing and boundary_smoothing > 0:
                data = ndimage.gaussian_filter(
                    data, boundary_smoothing, mode='mirror'
                )
#             contour = plt.contour(
#                 data, levels=[0.5], colors=boundary_color, **boundary_style
#             )

            # iterate over paths and simplify them
#             for c in contour.collections:
#                 for p in c.get_paths():
#                     p.simplify_threshold = .1
#                     p.vertices, p.codes = mpath.cleanup_path(
#                         p, None, True, None, False, 1., None, True
#                     )




if __name__ == "__main__":
    print('This file is intended to be used as a module.')
    print('This code serves as a test for the defined methods.')

    xs, ys = np.meshgrid(np.linspace(0, 10, 201), np.linspace(0, 10, 201))
    z = np.empty((201, 201, 2))
    z[:, :, 0] = np.sin(xs) > 0.5*np.cos(ys)
    z[:, :, 1] = np.cos(ys) > 0.2*np.sin(xs)

    z[range(100), range(100), 0] = np.nan

    make_contour_plot_hatched(
        z, ('r', 'g'),
        stripe_width=0.04, stripe_orientation=('\\', '/'),
        interpolation='none',
        boundary_style={'linewidths': 3}
    )
    plt.show()
