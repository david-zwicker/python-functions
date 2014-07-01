"""
Python module defining classes for creating more versatile figures

This code and information is provided 'as is' without warranty of any kind,
either express or implied, including, but not limited to, the implied
warranties of non-infringement, merchantability or fitness for a particular
purpose.
"""

from __future__ import division

import os
import shutil
import tempfile

import numpy as np


def numbers2latex(values, **kwargs):
    """
    Converts a list of numbers into a representation nicely displayed by latex.
    Additional parameters are passed on to `number2latex`
    """

    # apply function to all parts of a list recursively
    if hasattr(values, '__iter__'):
        return [numbers2latex(v, **kwargs) for v in values]
    else:
        return number2latex(values, **kwargs)


def number2latex(val, **kwargs):
    """ Converts a number into a representation nicely displayed by latex """

    # apply function to all parts of a potential list
    if hasattr(val, '__iter__'):
        print('Using the iterative version of `number2latex` is deprecated')
        return [number2latex(v, **kwargs) for v in val]

    # read parameters
    exponent_threshold = kwargs.pop('exponent_threshold', 3)
    add_dollar = kwargs.pop('add_dollar', False)
    precision = kwargs.pop('precision', None)

    # represent the input as mantissa + exponent
    val = float(val)
    if val == 0:
        exponent = 0
    else:
        exponent = int(np.log10(abs(val)))
        if exponent < 0:
            exponent -= 1
    mantissa = val/10**exponent

    # process these values further
    if precision is not None:
        mantissa = round(mantissa, precision)

    # distinguish different format cases
    if mantissa == 0:
        res = '0'

    elif abs(exponent) > exponent_threshold:

        # write mantissa
        res = "%g" % mantissa

        # handle special mantissa that can be omitted
        if res == '1':
            res = ''
        elif res == '-1':
            res = '-'
        elif res == '10':
            res = ''
            exponent += 1
        elif res == '-10':
            res = '-'
            exponent += 1
        else:
            res += r' \times '

        res += '10^{%d}' % (exponent)

    elif precision is not None:
        res = '%g' % round(val, precision - exponent)

    else:
        res = '%g' % val

    # check, whether to enclose the expression in dollar signs
    if add_dollar:
        res = '$%s$' % res

    return res


def tex2pdf(tex_source, outfile, use_pdflatex=True, verbose=False):

    if verbose:
        verbosity = ""
    else:
        verbosity = " &> /dev/null"

    outfile = os.path.abspath(outfile)

    # create temporary working directory
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp()

    # write TeX code to file
    f = open(tmp + "/document.tex", "w")
    f.write(tex_source)
    f.close()

    # create PDF
    os.chdir(tmp)
    if use_pdflatex:
        os.system("pdflatex document.tex" + verbosity)
    else: # use ordinary latex
        os.system("latex document.tex" + verbosity)
        os.system("dvips document.dvi" + verbosity)
        os.system("ps2pdf document.ps" + verbosity)

    # output resulting PDF
    shutil.copyfile(tmp + "/document.pdf", outfile)

    # house keeping
    os.system("rm -rf %s" % tmp)
    os.chdir(cwd)



if __name__ == "__main__":
    print('This file is intended to be used as a module.')
    print('This code serves as a test for the defined methods.')
    testvalues = (
        0.1, 1., 1.0001, -0.1, -1, 1e-8, 1e8,
        3.4567e-4, 3.4567e4, -3.4567e-4, -3.4567e4
    )
    for testval in testvalues:
        print("%s\t%s" % (testval, number2latex(testval, precision=3)))
