"""
Python module defining classes for creating more versatile figures

This code and information is provided 'as is' without warranty of any kind,
either express or implied, including, but not limited to, the implied
warranties of non-infringement, merchantability or fitness for a particular
purpose.
"""

from __future__ import division

import logging
import os
import pipes
from contextlib import contextmanager
import itertools

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mclr
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator


from latex_functions import number2latex

__all__ = [
    'invert_color',
    'FigureLatex', 'FigurePresentation',
    'figure_display', 'figure_file', 'figures'
]

GOLDEN_MEAN = 2/(np.sqrt(5) - 1)
INCHES_PER_PT = 1.0/72.27 # Convert pt to inch

# colors suitable for colorblinds
#COLOR_BLUE = '#0072B2'
COLOR_BLUE_SAFE = '#0673B7'
COLOR_ORANGE_SAFE = '#EFE342'
COLOR_GREEN_SAFE = '#009D73'
COLOR_RED_SAFE = '#D45F14'

# my nice colors
COLOR_BLUE = '#0673B7'
COLOR_ORANGE = '#FF7600'
COLOR_GREEN = '#00A919'
COLOR_RED = '#E6001C'

COLOR_LIST_STANDARD = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
COLOR_LIST_SAFE = [
    COLOR_BLUE_SAFE, COLOR_RED_SAFE, COLOR_GREEN_SAFE, COLOR_ORANGE_SAFE, 'k'
]
COLOR_LIST = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_RED, 'k']

# initialize the color converted and keep it as a global variable
color_converter = mclr.ColorConverter()


def invert_color(color):
    """ Returns the inverted value of a matplotlib color """
    # get the color value
    c = color_converter.to_rgba(color)
    # keep alpha value intact!
    return (1-c[0], 1-c[1], 1-c[2], c[3])


def shellquote(s):
    """ Quotes characters problematic for most shells """
    return pipes.quote(s)


def set_presentation_style_of_axis(axis, num_ticks=7, use_tex=True):
    """ private function setting a single axis to presentation style """

    # adjust tick formatter
    if use_tex:
        def apply_format(val, val_str):
            """ Helper function applying the format """
            return number2latex(val, add_dollar=True)
        axis.set_major_formatter(FuncFormatter(apply_format))
        if axis.get_scale() == 'linear':
            axis.set_minor_formatter(FuncFormatter(apply_format))
    else:
        axis.set_major_formatter(FormatStrFormatter('%g'))
        if axis.get_scale() == 'linear':
            axis.set_minor_formatter(FormatStrFormatter('%g'))

    # adjust the number of ticks
    if num_ticks == 0:
        axis.set_ticks([])
    elif axis.get_scale() == 'linear':
        axis.set_major_locator(MaxNLocator(num_ticks, steps=[1, 2, 5, 10]))


class FigureBase(mpl.figure.Figure):
    """ Extended version of a matplotlib figure """

    # factors used to determine the preferred number of ticks in each direction
    xtick_factor = 0.66
    ytick_factor = 1.

    def __init__(self, **kwargs):
        self.backend_old = None

        # read parameters
        self.transparent = kwargs.pop('transparent', None)
        self.verbose = kwargs.pop('verbose', False)

        backend = kwargs.pop('backend', None)
        safe_colors = kwargs.pop('safe_colors', False)
        fig_width_pt = kwargs.pop('fig_width_pt', 246.)
        aspect = kwargs.pop('aspect', None)
        dx = kwargs.pop('dx', None)
        dy = kwargs.pop('dy', None)
        num_ticks = kwargs.pop('num_ticks', None)

        # switch backend, if requested
        if backend is not None:
            self.backend_old = plt.get_backend()
            plt.switch_backend(backend)
            if backend.lower() != plt.get_backend().lower():
                logging.warn(
                    'Backend could not be switched from `%s` to `%s`',
                    plt.get_backend(), backend
                )

        # choose the color list used in this figure
        if safe_colors:
            self.colors = COLOR_LIST_SAFE
        else:
            self.colors = COLOR_LIST

        # set the number of ticks
        if num_ticks:
            if hasattr(num_ticks, '__iter__'):
                self.num_ticks = num_ticks[:2]
            else:
                self.num_ticks = (num_ticks, num_ticks)
        else:
            self.num_ticks = (None, None)

        # calculate the figure size
        if fig_width_pt is None:
            figsize = kwargs.pop('figsize', None)

        else:

            # calculate lengths
            if aspect is None:
                aspect = GOLDEN_MEAN

            if dx is None:
                dx = 4.*plt.rcParams['axes.labelsize']/fig_width_pt
            if not hasattr(dx, '__iter__' ):
                dx = (dx, 0.05)

            if dy is None:
                dy = 4.5*plt.rcParams['axes.labelsize']/fig_width_pt
            if not hasattr(dy, '__iter__' ):
                dy = (dy, 0.05)

            if self.verbose:
                logging.info('Parameter dx: %g, %g', *dx)
                logging.info('Parameter dy: %g, %g', *dy)

            fig_width = fig_width_pt*INCHES_PER_PT  # width in inches
            axes_width = fig_width*(1. - dx[0] - dx[1])
            axes_height = axes_width/aspect    # height in inches
            fig_height = axes_height/(1. - dy[0] - dy[1])
            figsize = [fig_width, fig_height]
            kwargs.pop('figsize', None)

        # setup the figure using the inherited constructor
        super(FigureBase, self).__init__(figsize=figsize, **kwargs)

        # setup the axes using the calculated dimensions
        self.ax = self.add_axes([dx[0], dy[0], 1-dx[0]-dx[1], 1-dy[0]-dy[1]])


    def get_color_iter(self, color=None):
        """
        Transforms the given color into a cycle or returns default colors.
        """
        if color is None:
            color = self.colors

        try:
            color_iter = itertools.cycle(color)
        except TypeError:
            color_iter = itertools.repeat(color)

        return color_iter


    def get_style_iter(self, color=True, dashes=None, extra=None):
        """
        Returns an iterator of various parameters controlling the style
        of plots.
        """

        # prepare the data
        if color in [True, None]:
            icolor = itertools.cycle(self.colors)
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
            """ helper function """
            while True:
                res = {'color': icolor.next()}
                if dashes is not None:
                    res['linestyle'] = idashes.next()
                if extra is not None:
                    res.update(extra)
                yield res

        return _style_generator()


    def invert_colors(self):
        """ Changes the colors of a figure to their inverted values """

        # keep track of the object that have been changed
        visited = set()

        def get_filter(name):
            """ construct a specific filter for `findobj` """
            return lambda x: hasattr(x, 'set_%s'%name) and \
                             hasattr(x, 'get_%s'%name)

        for o in self.findobj(get_filter('facecolor')):
            if o not in visited:
                o.set_facecolor(invert_color(o.get_facecolor()))
                if hasattr(o, 'set_edgecolor') and hasattr(o, 'get_edgecolor'):
                    o.set_edgecolor(invert_color(o.get_edgecolor()))
                visited.add(o)

        for o in self.findobj(get_filter('color')):
            if o not in visited:
                o.set_color(invert_color(o.get_color()))
                visited.add(o)

        # update canvas
        self.canvas.draw()


    @contextmanager
    def inverted_colors(self):
        """ Invert the colors and change them back after yielding """
        self.invert_colors()
        yield self
        self.invert_colors()


    def post_process(self, legend_frame=False):
        """
        Post process the image to adjust some things
        """

        # iterate over all axes
        for ax in self.axes:
            # adjust the ticks of the x-axis
            if self.num_ticks[0] is not None:
                num_ticks_x = self.num_ticks[0]
            else:
                num_ticks_x = self.xtick_factor*plt.rcParams['figure.figsize'][0]
            set_presentation_style_of_axis(ax.get_xaxis(), int(num_ticks_x))

            # adjust the ticks of the y-axis
            if self.num_ticks[1] is not None:
                num_ticks_y = self.num_ticks[1]
            else:
                num_ticks_y = self.ytick_factor*plt.rcParams['figure.figsize'][1]
            set_presentation_style_of_axis(ax.get_yaxis(), int(num_ticks_y))

            # adjust the legend
            legend = ax.get_legend()
            if legend is not None:
                legend.draw_frame(legend_frame)


    def savefig_pdf(self, filename, crop_pdf=False, **kwargs):
        """
        Saves a figure as a PDF file. If the filename ends with .eps or .ps,
        we first create a EPS or PS file and then convert it to PDF.
        """

        # prepare data
        filename, extension = os.path.splitext(filename)
        if extension == '':
            extension = '.pdf'
        file_pdf = filename + '.pdf'

        # save figure in the requested format
        self.savefig(
            filename + extension, transparent=self.transparent, **kwargs
        )
        if extension == '.ps':
            fmt = {
                'ps': shellquote(filename + '.ps'),
                'pdf': shellquote(file_pdf)
            }
            os.system('ps2pdf {ps} {pdf}.pdf'.format(**fmt))
        elif extension == '.eps':
            fmt = {
                'eps': shellquote(filename + '.eps'),
                'pdf': shellquote(file_pdf)
            }
            os.system('epspdf {eps} {pdf}'.format(**fmt))

        if crop_pdf:
            fmt = shellquote(file_pdf)
            os.system('pdfcrop {0} {0} &> /dev/null &'.format(fmt))

        return file_pdf


    def savefig_inverted(
            self, filename, background_facecolor=None,
            background_edgecolor=None, **kwargs
        ):
        """ Saves the figure to `filename` with inverted colors """

        rgb = color_converter.to_rgb

        if background_facecolor is None:
            bg_face = self.get_facecolor()
            if rgb(bg_face) == rgb(mpl.rcParamsDefault['figure.facecolor']):
                bg_face = 'k'
            else:
                bg_face = invert_color(bg_face)
        else:
            bg_face = background_facecolor

        if background_edgecolor is None:
            bg_edge = self.get_edgecolor()
            if rgb(bg_edge) == rgb(mpl.rcParamsDefault['figure.edgecolor']):
                bg_edge = 'none'
            else:
                bg_edge = invert_color(bg_edge)
        else:
            bg_edge = background_edgecolor

        # save inverted figure
        with self.inverted_colors():
            file_pdf = self.savefig_pdf(
                filename, facecolor=bg_face, edgecolor=bg_edge, **kwargs
            )
        return file_pdf



class FigureLatex(FigureBase):
    r""" Creates a latex figure of a certain width which should fit nicely
      The width must be given in pt and may be retrieved by
      \showthe\columnwidth or \the\columnwidth
    """

    def __init__(self, **kwargs):

        # read configuration
        font_size = kwargs.pop('font_size', 11)

        # setup all parameters
                # setup remaining parameters
        plt.rcParams.update({
          'axes.labelsize': font_size,
          'text.fontsize': font_size,
          'font.family': 'serif',
          'legend.fontsize': font_size,
          'xtick.labelsize': 0.9*font_size,
          'ytick.labelsize': 0.9*font_size,
          'text.usetex': True,
          'legend.loc': 'best',
          'font.serif': 'Computer Modern Roman, Times, Palatino, New Century '\
                        'Schoolbook, Bookman',
          'pdf.compression': 4,
        })

        # create figure using the inherited constructor
        super(FigureLatex, self).__init__(**kwargs)



class FigurePresentation(FigureBase):
    """ Creates a figure suitable for presentations
    """

    def __init__(self, **kwargs):

        # read configuration
        font_size = kwargs.pop('font_size', 11)

        # setup latex preamble
        preamble = \
            mpl.rcsetup.validate_stringlist(plt.rcParams['text.latex.preamble'])
        if r'\usepackage{sfmath}' not in preamble:
            preamble += [
                r'\sffamily',
                r'\usepackage{sfmath}',
                r'\renewcommand{\familydefault}{\sfdefault}'
            ]

        # setup all parameters
        plt.rcParams.update({
          'axes.labelsize': font_size,
          'text.fontsize': font_size,
          'font.family': 'sans-serif',
          'legend.fontsize': font_size,
          'xtick.labelsize': 0.9*font_size,
          'ytick.labelsize': 0.9*font_size,
          'text.usetex': True,
          'text.latex.preamble': preamble,
          'legend.loc': 'best',
          'font.sans-serif': 'Computer Modern Sans Serif, Bitstream Vera Sans,'\
            'Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica,'\
            'Avant Garde, sans-serif',
          'pdf.compression': 4,
        })

        # create figure using the inherited constructor
        super(FigurePresentation, self).__init__(**kwargs)



@contextmanager
def figure_display(
        FigureClass=None, post_process=True, legend_frame=False, **kwargs
    ):
    """ Provides a context manager for handling figures for display """

    if FigureClass is None:
        FigureClass = FigurePresentation

    # create figure
    fig = plt.figure(FigureClass=FigureClass, **kwargs)

    # return figure for plotting
    yield fig

    # show the figure
    if post_process:
        fig.post_process(legend_frame)
    plt.show()


@contextmanager
def figure_file(
        filename, FigureClass=None, crop_pdf=None, post_process=True,
        legend_frame=False, hold_figure=False, **kwargs
    ):
    """ Provides a context manager for handling figures for latex """

    if FigureClass is None:
        FigureClass = FigurePresentation

    # create figure
    fig = plt.figure(FigureClass=FigureClass, **kwargs)

    # return figure for plotting
    yield fig

    # save the figure to a file
    if post_process:
        fig.post_process(legend_frame)
    fig.savefig_pdf(filename=filename, crop_pdf=crop_pdf)
    if not hold_figure:
        plt.close(fig)


def figures(filename, **kwargs):
    """ Generator yielding two figure instances producing a latex and a
    presentation representation of a plot """

    # split filename to be able to insert content
    name, extension = os.path.splitext(filename)

    data = (('_latex', FigureLatex), ('_presentation', FigurePresentation))
    for style, cls in data:
        filename = name + style + extension
        with figure_file(filename, cls, **kwargs) as f:
            yield f


if __name__ == "__main__":
    print('This file is intended to be used as a module.')
    print('This code serves as a test for the defined methods.')

    test_x = np.linspace(0, 10, 100)

    # testplot with presentation style
    for fig in figures('test.pdf'):
        plt.plot(test_x, np.sin(test_x), "r", test_x, np.cos(test_x), "b")
        plt.xlabel("Coordinate $x$")
        plt.ylabel("f(x)")
        plt.title("Simple Plot")
        plt.legend(("sin(x)","cos(x)"))
