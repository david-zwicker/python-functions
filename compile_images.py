#!/usr/bin/env python
"""
Small script which combines images given on the command line into a single
PDF file.

Basic Usage:
    ./compile_images -o output.pdf input1.eps input2.eps

Run `./compile_images --help` for more details.

@copyright: David Zwicker <dzwicker@seas.harvard.edu>
@date:      2014-07-10
"""

import os
import tempfile
import shutil
from optparse import OptionParser

# define options of the command line
parser = OptionParser("compile_images.py [options] image1 [image2]...")
parser.add_option(
    "-2", "--twocolumns", action="store_true", dest="twocolumns", default=False,
    help="use two columns for images"
)
parser.add_option(
    "-c", "--caption", dest="caption", help="add a caption to the PDF",
    metavar="TITLE"
)
parser.add_option(
    "-o", "--output", dest="output", help="write resulting PDF to FILE",
    metavar="FILE"
)
parser.add_option(
    "-i", "--information", dest="infofile", help="includes TeX-code from FILE",
    metavar="FILE"
)
parser.add_option(
    "-t", "--tex", action="store_true", dest="tex", default=False,
    help="output TEX source code"
)
parser.add_option(
    "-n", "--no-title", action="store_true", dest="no_title", default=False,
    help="prevents the addition of a title below the image"
)
parser.add_option(
    "-q", "--quiet", action="store_false", dest="verbose", default=True,
    help="don't print status messages to stdout or stderr"
)
#parser.add_option( "-w", "--wide",
#                   action="store_true", dest="wide", default=False,
#                   help="use wide format with decreased margins"
#                 )

# parse options from the command line
(options, args) = parser.parse_args()

# output help, if no arguments are given
if len(args) == 0:
    parser.print_help()
    exit()

# set some initial parameters depending on the parsed options
if options.output is None:
    outfile = None
    options.verbose = False
else:
    outfile = os.path.abspath(options.output)

if options.verbose:
    verbosity = ""
else:
    verbosity = " &> /dev/null"

if options.twocolumns:
    TITLE_CHARACTERS = 45
    WIDTH_STRING = "0.45\\textwidth"
else:
    TITLE_CHARACTERS = 80
    WIDTH_STRING = "\\textwidth"


# initialize data
EPS = False
PDF = False

# create TeX code
if options.verbose:
    print("Scanning images...")

# check whether a caption is requested
if options.caption is None:
    s = ""
else:
    s = """\\begin{center}
\\section*{%s}
\\end{center}
\\thispagestyle{empty}
""" % options.caption

# add possible extra information
if options.infofile is not None:
    file_handle = open(options.infofile, 'r')
    s += r'\begin{minipage}{\textwidth}' \
        + file_handle.read() \
        + r'\end{minipage}\\'
    file_handle.close()

# run through all remaining options, which are the filenames of the images
for k, filename in enumerate(args):

    # check the image file
    img = os.path.abspath(filename)
    if os.path.exists(img):
        title = img
        filename = img
        ext = os.path.splitext(img)[1]
        if ext == ".eps":
            EPS = True
        elif ext == ".pdf":
            PDF = True
    elif os.path.exists(img + ".eps"):
        title = img
        filename = "%s.eps" % img
        EPS = True
    elif os.path.exists(img + ".pdf"):
        title = img
        filename = "%s.pdf" % img
        PDF = True
    else:
        if options.verbose:
            print("File '%s' does not exist and will be ignored" % img)
        continue

    # handle the title
    if options.no_title:
        title_str = ""
    else:
        # strip title string if it is too long
        if len(title) > TITLE_CHARACTERS:
            title = "...%s" % title[3-TITLE_CHARACTERS:]
        title_str = "\\verb\"%s\"" % title

    # add the necessary tex-code
    s += """\\begin{minipage}{%(width)s}
    \\includegraphics[width=\\textwidth]{%(filename)s}
    %(title)s
    \\end{minipage}
    """ % {'width':WIDTH_STRING, 'filename':filename, 'title':title_str}

    # handle the space between figures
    if options.twocolumns and k % 2 == 0:
        # we have to place the figure next to another one
        s += "\\hfill"


# define latex document structure
tex = """\\documentclass[10pt]{article}
\\usepackage[top=2cm, bottom=2cm, left=1cm, right=1cm, a4paper]{geometry}
\\usepackage{graphicx} %% include graphics
\\usepackage{grffile}  %% support spaces in filenames
\\begin{document}%%
%s
\\vfill
\\end{document}
""" % s

# decide what to output
if options.tex:
    # output TeX code
    print(tex)
else:
    if options.verbose:
        print("Compiling PDF file...")

    # create temporary working directory
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp()

    # write TeX code to file
    f = open(tmp + "/img.tex", "w")
    f.write(tex)
    f.close()

    # create PDF
    os.chdir(tmp)
    if PDF and EPS:
        print("Can't compile images, since both EPS and PDF files are given.")
    elif PDF: # images are PDF documents
        os.system("pdflatex img.tex" + verbosity)
    else: # images are EPS documents
        os.system("latex img.tex" + verbosity)
        os.system("dvips img.dvi" + verbosity)
        os.system("ps2pdf img.ps" + verbosity)

    # output resulting PDF
    if outfile is None:
        f = open(tmp + "/img.pdf")
        print(f.read())
        f.close()
    else:
        shutil.copyfile(tmp + "/img.pdf", outfile)

    # house keeping
    os.system("rm -rf %s" % tmp)
    os.chdir(cwd)
