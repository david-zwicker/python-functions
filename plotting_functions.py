""" This file contains several functions useful for plotting figures using
matplotlib """

from __future__ import division

import pipes
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mclr
from matplotlib.patches import Polygon
from matplotlib.ticker import Locator

from figure_presets import set_presentation_style_of_axis

# import to make functions available in this namespace
from latex_functions import number2latex, numbers2latex  # @UnusedImport

GOLDEN_MEAN = 2/(np.sqrt(5) - 1)

# nice colors
COLOR_BLUE_OLD = '#0673B7'
COLOR_ORANGE_OLD = '#FF7600'
COLOR_GREEN_OLD = '#00A919'
COLOR_RED_OLD = '#E6001C'

# colors suitable for color blind people
#COLOR_BLUE = '#0072B2'
COLOR_BLUE_SAFE = '#0673B7'
COLOR_ORANGE_SAFE = '#EFE342'
COLOR_GREEN_SAFE = '#009D73'
COLOR_RED_SAFE = '#D45F14'

COLOR_BLUE = '#0673B7'
COLOR_ORANGE = '#FF7600'
COLOR_GREEN = '#00A919'
COLOR_RED = '#E6001C'

# COLOR_LIST_SAFE = [
#     COLOR_BLUE_SAFE, COLOR_RED_SAFE, COLOR_GREEN_SAFE, COLOR_ORANGE_SAFE, 'k'
# ]

# this list has been taken from
# Wong. Color blindness. Nat Methods (2011) vol. 8 (6) pp. 441
COLOR_LIST_SAFE = [
    '#0072B2', # Blue
    '#D55E00', # Vermillion
    '#009E73', # Bluish green
    '#E69F00', # Orange
    '#56B4E9', # Sky blue
    '#F0E442', # Yellow
    '#CC79A7', # Reddish purple
    'k'        # Black
]
COLOR_LIST = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_RED, 'k']



def shellquote(s):
    """ Quotes characters problematic for most shells """
    return pipes.quote(s)



def set_presentation_style(fig=None, legend_frame=False, axes=None, scale=1.):
    """ Changes the style of a figure to be useful for presentations """

    # secondary scale factor
    scale2 = np.sqrt(scale)

    # get the right figure handle
    if fig is None:
        fig = plt.gcf()

    # compile the list of axes
    if axes is None:
        axes = set()
    else:
        axes = set(axes)
    axes |= set(fig.axes)

    # run through all axes objects
    for ax in axes:

        # adjust the axes
        set_presentation_style_of_axis(ax.get_xaxis())
        set_presentation_style_of_axis(ax.get_yaxis())

        # adjust the frame around the image
        for spine in ax.spines.values():
            spine.set_linewidth(2.*scale)

        # adjust all lines within the axes
        for line in ax.get_lines():
            line.set_linewidth(3./2.*scale*line.get_linewidth())

        # adjust all text objects
        for text in ax.findobj(plt.Text):
            text.set_fontname('Helvetica')
            text.set_fontsize(16*scale2)

        # adjust the tick padding
        ax.tick_params(
            pad=6*scale, width=2*scale, length=7*scale
        )

        # adjust the legend, if there is any
        legend = ax.get_legend()
        if legend is not None:
            legend.draw_frame(legend_frame)
            for line in legend.get_lines():
                line.set_linewidth(2.*scale)


    # redraw figure
    fig.canvas.draw()



def set_axis_color(ax=None, axis='y', color='r'):
    """ Changes the color of an axis including the ticks and the label """

    if ax is None:
        ax = plt.gca()

    ax.tick_params(axis=axis, which='both', color=color, labelcolor=color)



def get_color_scheme(base_color, num=4, spread=1.):
    """ Distributes num colors around the color wheel starting with a base
    color and converting the fraction `spread` of the circle """
    base_rgb = mclr.colorConverter.to_rgb(base_color)
    base_rgb = np.reshape(np.array(base_rgb), (1, 1, 3))
    base_hsv = mclr.rgb_to_hsv(base_rgb)[0, 0]
    res_hsv = np.array([[
        ((base_hsv[0] + dh) % 1., base_hsv[1], base_hsv[2])
        for dh in np.linspace(-0.5*spread, 0.5*spread, num, endpoint=False)
    ]])
    return mclr.hsv_to_rgb(res_hsv)[0]



def get_color_iter(color=None):
    """
    Transforms the given color into a cycle or returns default colors
    """
    if color is None:
        color = COLOR_LIST

    try:
        color_iter = itertools.cycle(color)
    except TypeError:
        color_iter = itertools.repeat(color)

    return color_iter

plot_colors = get_color_iter



def get_style_iter(color=True, dashes=None, extra=None):
    """ Returns an iterator of various parameters controlling the style
    of plots """

    # prepare the data
    if color in [True, None]:
        icolor = itertools.cycle(COLOR_LIST)
    elif color == False:
        icolor = itertools.repeat('k')
    else:
        icolor = itertools.cycle(color)

    if dashes in [False, None]:
        idashes = itertools.repeat('-')
    elif dashes == True:
        idashes = itertools.cycle(['-', '--', ':', '-.' ])
    else:
        idashes = itertools.cycle(dashes)

    # function yielding the iterator
    def _style_generator():
        """ Helper function """
        while True:
            res = {'color': next(icolor)}
            if dashes is not None:
                res['linestyle'] = next(idashes)
            if extra is not None:
                res.update(extra)
            yield res

    return _style_generator()

plot_styles = get_style_iter



def get_colormap(colors='rgb'):
    """ builds a segmented colormap with the color sequence given
    as a string """
    COLORS = {
        'r': '#E6001C', 'g': '#00A919', 'b': '#0673B7',
        'w': '#FFFFFF', 'k': '#000000'
    }

    return mclr.LinearSegmentedColormap.from_list(
        colors, [COLORS[c] for c in colors]
    )



def blend_colors(color, bg='w', alpha=0.5):
    """
    Blends two colors using a weight. Can be used for faking alpha
    blending
    """
    if not hasattr(blend_colors, 'to_rgb'):
        blend_colors.to_rgb = mclr.ColorConverter().to_rgb

    return alpha*np.asarray(blend_colors.to_rgb(color)) \
        + (1 - alpha)*np.asarray(blend_colors.to_rgb(bg))



def reordered_legend(order=None, ax=None, *args, **kwargs):
    """
    Reorders the legend of an axis
    """
    if ax is None:
        ax = plt.gca()

    ax.legend(*args, **kwargs)
    if order is not None:
        handles, labels = ax.get_legend_handles_labels()

        handles = np.asarray(handles)
        labels = np.asarray(labels)

        ax.legend(handles[order], labels[order], *args, **kwargs)



def errorplot(x, y, yerr=None, fmt='', **kwargs):
    """
    Creates an error plot in which y-errors are represented by a band instead
    of individual errorbars
    
    `subsample` allows to plot only a fraction of the actual data points in the
        plot of the mean, while all data points are used for the envelope
        showing the errorbars 
    """
    label = kwargs.pop('label', None)
    subsample = kwargs.pop('subsample', 1)
    has_error = (yerr is not None)
    
    # plot the mean
    if fmt != 'none':
        if has_error:
            line_label = None
        else:
            line_label = label
        line = plt.plot(x[::subsample], y[::subsample], fmt, label=line_label,
                        **kwargs)[0]
        color = kwargs.pop('color', line.get_color())
    else:
        line = None
        color = kwargs.pop('color', None)
        
    # plot the deviation
    if has_error:
        alpha = kwargs.pop('alpha', 0.3)
        kwargs.pop('ls', None)  #< ls only applies to centerline
        
        y = np.asarray(y)
        yerr = np.asarray(yerr)
        
        shape_err = plt.fill_between(x, y - yerr, y + yerr, color=color,
                                     edgecolors='none', alpha=alpha, 
                                     label=label, **kwargs)
    else:
        shape_err = None

    return line, shape_err



def scatter_barchart(data, labels=None, ax=None, barwidth=0.7, color=None):
    """
    Creates a plot of groups of data points, where the individual points
    are represented by a scatter plot and the mean for each group is
    depicted as a bar in a histogram style.
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    if not hasattr(data[0], '__iter__'):
        data = [data]

    color = get_color_iter(color)
    w = 0.5*barwidth

    # make the histogram
    x = np.arange(len(data)) - w
    y = [np.mean(vals) for vals in data]
    plt.bar(
        x, y, width=barwidth, color='white', edgecolor='black', linewidth=1.
    )

    # add the scatter plot on top
    for gid, ys in enumerate(data):
        xs = np.linspace(gid-0.9*w, gid+0.9*w, len(ys))
        ax.plot(
            xs, ys, 'ko', markersize=8, markerfacecolor=next(color),
            markeredgewidth=0., markeredgecolor='white'
        )

    # ensure that the space between the bars equals the margin to the frame
    ax.set_xlim(w - 1., len(data) - w)

    # set labels
    if labels is not None:
        plt.xticks(range(len(labels)), labels)
    else:
        plt.xticks(range(len(data)))



def contour_to_hatched_patches(
        cntrset, hatch_colors, hatch_patterns, remove_contour=True
    ):
    """ Function turning a filled contour plot into an equivalent one
    using hatches to show areas.
    Code has been taken from StackOverflow!
    """
    from itertools import cycle
    from matplotlib.patches import PathPatch

    ax = plt.gca()
    patches_list = []
    for pathcollection in cntrset.collections:
        patches_list.append([PathPatch(p) for p in  pathcollection.get_paths()])
        if remove_contour:
            pathcollection.remove()

    hatch_colors = cycle(hatch_colors)
    hatch_patterns = cycle(hatch_patterns)

    for patches, _, hp in zip(patches_list, hatch_colors, hatch_patterns):
        for p in patches:
            p.set_fc("none")
            p.set_ec("k")
            p.set_hatch(hp)
            ax.add_patch(p)



def plot_length(xs, ys, w=0.1, **kwargs):
    """ Plots a double arrow between the two given coordinates """

    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = plt.gca()

    # set parameters of the arrow
    arrowparams = {
        'head_width':2*w, 'head_length':w,
        'length_includes_head':True, 'shape':'full',
        'head_starts_at_zero':False
    }
    arrowparams.update(kwargs)

    # plot two arrows to mimic double arrow
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    ax.arrow(xs[0], ys[0],  dx,  dy, **arrowparams)
    ax.arrow(xs[1], ys[1], -dx, -dy, **arrowparams)



def logspace(start, stop, *args, **kwargs):
    """ Distributes `num` numbers logarithmically between `start` and `stop`"""
    return np.exp(np.linspace(np.log(start), np.log(stop), *args, **kwargs))



def log_slope_indicator(
        xmin=1., xmax=2., factor=None, ymax=None,
        exponent=1., label_x='', label_y='',
        space=15, loc='lower', ax=None, **kwargs
    ):
    """
    Function adding a triangle to axes `ax`. This is useful for indicating
    slopes in log-log-plots. `xmin` and `xmax` denote the x-extend of the
    triangle. The y-coordinates are calculated according to the formula
        y = factor*x**exponent
    If supplied, the texts `label_x` and `label_y` are put next to the
    catheti. The parameter `loc` determines whether the catheti are
    above or below the diagonal. Additionall kwargs can be used to
    set the style of the triangle
    """

    # prepare
    if ax is None:
        ax = plt.gca()
    lower = (loc == 'lower') != (exponent < 0.)

    if ymax is not None:
        factor = ymax/max(xmin**exponent, xmax**exponent)

    if factor is None:
        factor = 1.

    # get triangle coordinates
    y = factor*np.array((xmin, xmax))**exponent
    if lower:
        pts = np.array([[xmin, y[0]], [xmax, y[0]], [xmax, y[1]]])
    else:
        pts = np.array([[xmin, y[0]], [xmax, y[1]], [xmin, y[1]]])

    # add triangle to axis
    if not( 'facecolor' in kwargs or 'fc' in kwargs ):
        kwargs['facecolor'] = 'none'
    p = Polygon(pts, closed=True, **kwargs)
    ax.add_patch(p)

    # labels
    xt = np.exp(0.5*(np.log(xmin) + np.log(xmax)))
    #dx = (xmax/xmin)**0.1
    yt = np.exp(np.log(y).mean())
    #dy = (y[1]/y[0])**0.1
    sgn = np.sign(exponent)
    if lower:
        ax.annotate(
            label_x, xy=(xt, y[0]), xytext=(0, -sgn*space),
            textcoords='offset points', size='x-small',
            horizontalalignment='center',
            verticalalignment='top'
        )
        ax.annotate(
            label_y, xy=(xmax, yt), xytext=(space, 0),
            textcoords='offset points', size='x-small',
            horizontalalignment='right',
            verticalalignment='center'
        )

    else:
        ax.annotate(
            label_x, xy=(xt, y[1]), xytext=(0, sgn*space),
            textcoords='offset points', size='x-small',
            horizontalalignment='center',
            verticalalignment='bottom'
        )
        ax.annotate(
            label_y, xy=(xmin, yt), xytext=(-space, 0),
            textcoords='offset points', size='x-small',
            horizontalalignment='left',
            verticalalignment='center'
        )



def plot_masked(x, y, mask, *args, **kwargs):
    """
    plots a line given by points x, y using a mask
    if `mask` is NaN, no line is drawn, if the mask evaluates to True,
        a solid line is used, otherwise a dotted line is drawn.
    """
    label = kwargs.pop('label', None)
    close_gaps = kwargs.pop('close_gaps', False)
    di = 1 if close_gaps else 0

    start = 0
    res = []
    for end in range(1, len(x)):
        if mask[end] != mask[start]:
            if np.isnan(mask[start]):
                pass
            elif mask[start]:
                res.append(plt.plot(
                    x[start:end+di], y[start:end+di],
                    '-', label=label, *args, **kwargs
                ))
                label = None
            else:
                res.append(plt.plot(
                    x[start:end+di], y[start:end+di], ':', *args, **kwargs
                ))
            start = end

    # print last line
    style = '-' if mask[start] else ':'
    res.append(
        plt.plot(x[start:], y[start:], style, label=label, *args, **kwargs)
    )
    return res



def plot_hist_logscale(data, bins=10, data_range=None, **kwargs):
    """ plots a histogram with logarithmic bin widths """
    # extract the positive part of the data
    data_pos = data[data > 0]
    
    if data_range is None:
        data_range = (data_pos.min(), data_pos.max())
    
    # try determining the bins
    try:
        bins = logspace(data_range[0], data_range[1], bins + 1)
        print(bins)
    except TypeError:
        # `bins` might have been a numpy array already
        pass
    
    res = plt.hist(data, bins=bins, **kwargs)
    plt.xscale('log')
    return res



def get_hatched_image(values, stripe_width=0.05, orientation='/'):
    """
    Takes three dimensional boolean data and projects out the last dimension,
    by using hatched regions to symbolize parts, where several variables are
    true
    """

    if stripe_width < 1:
        stripe_width = int(stripe_width*max(values.shape[:2]))

    # build orientation list
    dimensions = len(values[0, 0, :])
    if hasattr(orientation, '__iter__'):
        orientations = itertools.cycle(orientation)
        orientations = list(itertools.islice(orientations, dimensions))
    else:
        orientations = [orientation]*dimensions

    # convert the values and calculate the number of values
    values = np.atleast_3d(values)
    nums = values.sum(2).astype(int)

    res = np.zeros(values.shape[:2])
    for x, y in np.ndindex(*res.shape):
        if nums[x, y] == 0:
            res[x, y] = -1
        else:
            orientation = orientations[nums[x, y] - 1]
            # choose a color index based on the current stripe
            if orientation == '\\':
                i = ((x - y) % stripe_width)*nums[x, y]//stripe_width
            elif orientation == '-':
                i = (x % stripe_width)*nums[x, y]//stripe_width
            elif orientation == '/':
                i = ((x + y) % stripe_width)*nums[x, y]//stripe_width
            elif orientation == '|':
                i = (y % stripe_width)*nums[x, y]//stripe_width
            else:
                raise ValueError(
                    'Allowed stripe orientation values: /, -, \\,  |'
                )
            # choose the color from the current values
            res[x, y] = np.nonzero(values[x, y, :])[0][i]

    return res



class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))



if __name__ == "__main__":
    print('This file is intended to be used as a module.')
    print('This code serves as a test for the defined methods.')

    tests = (
#         'safe_colors',
#         'axis_color',
#         'scatter_barchart',
#         'presentation_style',
#         'log_slope_indicator',
#         'hatched_image',
#         'figures',
        'plot_hist_logscale',
    )

    if 'safe_colors' in tests:
        plt.clf()

        for i, color in enumerate(COLOR_LIST_SAFE):
            plt.axhspan(0.5-i, -i, color=color)

        plt.show()

    if 'axis_color' in tests:
        plt.clf()

        test_x = np.linspace(0, 5, 100)
        # testplot with colorful axis

        plt.plot(test_x, np.sin(test_x), c='r')
        plt.xlabel("x")
        plt.ylabel("sin(x)")

        set_axis_color(axis='x', color='b')
        set_axis_color(axis='y', color='r')

        ax2 = plt.twinx()
        plt.plot(test_x, np.cos(test_x), c='g')
        plt.ylabel("cos(x)")
        set_axis_color(ax2, axis='y', color='g')

        plt.show()

    if 'scatter_barchart' in tests:
        # testplot with scatter_barchart
        plt.clf()
        test_data = np.zeros((5, 10))
        for i in range(5):
            test_data[i, :] = i + np.random.rand(10)

        scatter_barchart(test_data, labels=('a', 'b', 'c', 'd', 'e'))

        plt.show()

    if 'presentation_style' in tests:
        test_x = np.linspace(0, 5, 100)

        # testplot with presentation style
        plt.clf()
        plt.plot(test_x, np.sin(test_x), "r", test_x, np.cos(test_x), "b")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Simple Plot")
        plt.legend(("sin(x)","cos(x)"))

        # apply presentation style to plot
        set_presentation_style()

        plt.show()

    if 'log_slope_indicator' in tests:
        test_x = logspace(1, 1000, 20)
        test_y = test_x**2
        test_y *= (1 + 0.1*np.random.randn(20))

        plt.loglog(test_x, test_y, '+')

        log_slope_indicator(
            xmin=10, xmax=100, factor=0.5, exponent=2.,
            label_x='1', label_y='2', ec='red'
        )
        log_slope_indicator(
            xmin=100, xmax=300, factor=2., exponent=2.,
            label_x='1', label_y='2', loc='upper'
        )

        plt.show()

    if 'hatched_image' in tests:
        test_x, test_y = np.meshgrid(np.linspace(0, 10, 201),
                                     np.linspace(0, 10, 201))
        z = np.empty((201, 201, 2), np.bool)
        z[:, :, 0] = np.sin(test_x) > 0
        z[:, :, 1] = np.cos(test_y) > 0

        img = get_hatched_image(z, stripe_width=12, orientation='\\')
        plt.imshow(img)
        plt.show()
        
    if 'plot_hist_logscale' in tests:
        test_data = np.exp(np.random.uniform(0, 10, 1000))
        plot_hist_logscale(test_data)
        plt.show()
        