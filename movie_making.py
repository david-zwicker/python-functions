"""
Python module defining a class for creating movies of matplotlib figures.

This code and information is provided 'as is' without warranty of any kind,
either express or implied, including, but not limited to, the implied
warranties of non-infringement, merchantability or fitness for a particular
purpose.
"""

from functools import partial
import shutil
import subprocess
import tempfile
import matplotlib as mpl
import matplotlib.pyplot as plt


def invert_color(color):
    """ Returns the inverted value of a matplotlib color """
    # get the color value
    c = invert_color.cc.to_rgba(color)
    # keep alpha value intact!
    return (1-c[0], 1-c[1], 1-c[2], c[3])

# initialize the color converted and keep it as a static variable
invert_color.cc = mpl.colors.ColorConverter()


def invert_colors(fig):
    """ Changes the colors of a figure to their inverted values """

    # keep track of the object that have been changed
    visited = set()

    def get_filter(name):
        """ construct a specific filter for `findobj` """
        return lambda x: hasattr(x, 'set_%s'%name) and hasattr(x, 'get_%s'%name)

    for o in fig.findobj(get_filter('facecolor')):
        if o not in visited:
            o.set_facecolor(invert_color(o.get_facecolor()))
            if hasattr(o, 'set_edgecolor') and hasattr(o, 'get_edgecolor'):
                o.set_edgecolor(invert_color(o.get_edgecolor()))
            visited.add(o)

    for o in fig.findobj(get_filter('color')):
        if o not in visited:
            o.set_color(invert_color(o.get_color()))
            visited.add(o)


class Movie(object):
    """ Class for creating movies from matplotlib figures using ffmpeg """

    def __init__(self,
            width=None, filename=None, inverted=False, verbose=True,
            framerate=None
        ):
        self.width = width          #< pixel width of the movie
        self.filename = filename    #< filename used to save the movie
        self.inverted = inverted    #< colors inverted?
        self.verbose = verbose      #< verbose encoding information?
        self.framerate = framerate  #< framerate of the movie

        # internal variables
        self.recording = False
        self.tempdir = None
        self.frame = 0
        self._start()


    def __del__(self):
        self._end()


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.filename is not None:
            self.save(self.filename)
        self._end()
        return False


    def _start(self):
        """ initializes the video recording """
        # create temporary directory for the image files of the movie
        self.tempdir = tempfile.mkdtemp(prefix='movie_')
        self.frame = 0
        self.recording = True


    def _end(self):
        """ clear up temporary things if necessary """
        if self.recording:
            shutil.rmtree(self.tempdir)
            self.recording = False


    def clear(self):
        """ delete current status and start from scratch """
        self._end()
        self._start()


    def _add_file(self, save_function):
        """
        Adds a file to the current movie
        """
        if not self.recording:
            raise ValueError('Movie is not initialized.')

        save_function("%s/frame_%09d.png" % (self.tempdir, self.frame))
        self.frame += 1


    def add_image(self, image):
        """
        Adds the data of a PIL image as a frame to the current movie.
        """
        if self.inverted:
            try:
                import ImageOps
            except ImportError:
                from PIL import ImageOps
            image_inv = ImageOps.invert(image)
            self._add_file(image_inv.save)
        else:
            self._add_file(image.save)


    def add_array(self, data, colormap=None):
        """
        Adds the data from the array as a frame to the current movie.
        The array is assumed to be scaled to [0,  1].
        (0, 0) lies in the upper left corner of the image.
        The first axis extends toward the right, the second toward the bottom
        """
        # get colormap
        if colormap is None:
            import matplotlib.cm as cm
            colormap = cm.gray

        # produce image
        try:
            import Image
        except ImportError:
            from PIL import Image
        import numpy as np
        grey_data = colormap(np.clip(data.T, 0, 1))
        im = Image.fromarray(np.uint8(grey_data*255))

        # save image
        self.add_image(im)


    def add_figure(self, fig=None):
        """ adds the figure `fig` as a frame to the current movie """
        if fig is None:
            fig = plt.gcf()

        if self.width is None:
            dpi = None
        else:
            dpi = self.width/fig.get_figwidth()

        # save image
        if self.inverted:
            invert_colors(fig)
            save_function = partial(
                fig.savefig,
                dpi=dpi, edgecolor='none',
                facecolor=invert_color(fig.get_facecolor())
            )
            self._add_file(save_function)
            invert_colors(fig)
        else:
            save_function = partial(fig.savefig, dpi=dpi)
            self._add_file(save_function)


    def save_frames(self, filename_pattern='./frame_%09d.png', frames='all'):
        """ saves the given `frames` as images using the `filename_pattern` """
        if not self.recording:
            raise ValueError('Movie is not initialized.')

        if 'all' == frames:
            frames = range(self.frame)

        for f in frames:
            shutil.copy(
                "%s/frame_%09d.png" % (self.tempdir, f),
                filename_pattern % f
            )


    def save(self, filename, extra_args=None):
        """ convert the recorded images to a movie using ffmpeg """
        if not self.recording:
            raise ValueError('Movie is not initialized.')

        # set parameters
        if extra_args is None:
            extra_args = []
        if self.framerate is not None:
            extra_args += ["-r", self.framerate]
        if filename is None:
            filename = self.filename

        # construct the call to ffmpeg
        # add the `-pix_fmt yuv420p` switch for compatibility reasons
        #     -> http://ffmpeg.org/trac/ffmpeg/wiki/x264EncodingGuide
        args = ["ffmpeg"]
        if extra_args:
            args += extra_args
        args += [
            "-y", # don't ask questions
            "-f", "image2", # input format
            "-i", "%s/frame_%%09d.png" % self.tempdir, # input data
            "-pix_fmt", "yuv420p", # pixel format for compatibility
            "-b:v", "1024k", # high bit rate for good quality
            filename # output file
        ]

        # spawn the subprocess and capture its output
        p = subprocess.Popen(args, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        out = p.stdout.read()
        err = p.stderr.read()

        # check if error occurred
        if p.wait():
            print(out)
            print(err)
            raise RuntimeError('An error occurred while producing the movie.')

        # do output anyway, when verbosity is requested
        if self.verbose:
            print(out)
            print(err)



def test_movie_making():
    """ Simple test code for movie making """

    try:
        # try python2 version
        filename = raw_input('Choose a file name: ')
    except NameError:
        # python3 fallback
        filename = input('Choose a file name: ')

    import numpy as np

    # prepare data
    x = np.linspace(0, 10, 100)
    lines, = plt.plot(x, np.sin(x))
    plt.ylim(-1, 1)

    with Movie(filename=filename) as movie:
        for k in range(30):
            lines.set_ydata(np.sin(x + 0.1*k))
            movie.add_frame()



if __name__ == "__main__":
    print('This file is intended to be used as a module.')
    print('This code serves as a test for the defined methods.')

    test_movie_making()
