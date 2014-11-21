'''
Created on Nov 21, 2014

@author: zwicker
'''

from __future__ import division

import math

import numpy as np
import scipy.stats
from scipy.optimize import newton
from scipy.integrate import quad
from scipy.special import gamma
from scipy.misc import factorial  # @UnresolvedImport


PI2 = 2*np.pi


def average_angles(data, period=PI2):
    """ averages a list of cyclic values (angles by default)
    data  The list of angles
    per   The period of the angular variables
    """
    data = np.asarray(data)    
    if period is not PI2:
        data *= PI2/period
    data = math.atan2(np.sin(data).sum(), np.cos(data).sum())
    if period is not PI2:
        data *= period/PI2
    return data



def logspace(start, end, num=None):
    """ Returns an ordered sequence of `num` numbers from `start` to `end`, which
        are spaced logarithmically """

    if num is None:
        return np.logspace(np.log(start), np.log(end), base=np.e)
    else:
        return np.logspace(np.log(start), np.log(end), num, base=np.e)



def random_log_uniform(v_min, v_max, size):
    """ returns random variables that a distributed uniformly in log space """
    log_min, log_max = np.log(v_min), np.log(v_max)
    res = np.random.uniform(log_min, log_max, size)
    return np.exp(res)



def sampling_distribution_std(x, std, num):
    """
    Sampling distribution of the standard deviation, see
    http://en.wikipedia.org/wiki/Chi_distribution
    """

    if x < 0:
        return 0

    scale = np.sqrt(num)/std
    x *= scale

    res = 2**(1 - 0.5*num)*x**(num - 1)*np.exp(-0.5*x*x)/gamma(0.5*num)

    return res*scale



def sampling_distribution_cv(x, cv, num):
    """
    Sampling distribution of the coefficient of variation, see
    W. A. Hendricks and K. W. Robey. The Sampling Distribution of the
    Coefficient of Variation. Ann. Math. Statist. 7, 129-132 (1936).
    """
    if x < 0:
        return 0
    x2 = x*x

    # calculate the sum
    res = sum(
        factorial(num - 1)*gamma(0.5*(num - i))/factorial(num - 1 - i)/factorial(i)*
        num**(0.5*i)/(2**(0.5*i)*cv**i)/((1 + x2)**(0.5*i))
        for i in range(1 - num%2, num, 2)
    )

    # multiply by the other terms
    res *= 2./(np.sqrt(np.pi)*gamma(0.5*(num-1)))
    res *= x**(num-2)/((1 + x2)**(0.5*num))
    res *= np.exp(-0.5*num/(cv**2)*x2/(1 + x2))

    return res



def confidence_interval(value, distribution=None, args=None, guess=None, confidence=0.95):
    """ returns the confidence interval of `value`, if its value was estimated
    `num` times from the `distribution` """

    if guess is None:
        guess = 0.1*value
    if distribution is None:
        distribution = sampling_distribution_std

    # create partial function
    distr = lambda y: distribution(y, value, *args)

    def rhs(x):
        """ integrand """
        return confidence - quad(distr, value-x, value+x)[0]

    res = newton(rhs, guess, tol=1e-4)
    return res



def confidence_interval_mean(std, num, confidence=0.95):
    """ calculates the confidence interval of the mean given a standard
    deviation and a number of observations, assuming a normal distribution """
    sem = std/np.sqrt(num) # estimate of the standard error of the mean

    # get confidence interval from student-t distribution
    factor = scipy.stats.t(num - 1).ppf(0.5 + 0.5*confidence)

    return factor*sem



def confidence_interval_std(std, num, confidence=0.95):
    """ calculates the confidence interval of the standard deviation given a 
    standard deviation and a number of observations, assuming a normal
    distribution """
    c = scipy.stats.chi(num - 1).ppf(0.5 + 0.5*confidence)
    lower_bound = np.sqrt(num - 1)*std/c

    c = scipy.stats.chi(num - 1).ppf(0.5 - 0.5*confidence)
    upper_bound = np.sqrt(num - 1)*std/c

    return 0.5*(upper_bound - lower_bound)



def estimate_mean(data, confidence=0.95):
    """ estimate mean and the associated standard error of the mean of a
    data set """
    mean = data.mean()
    std = data.std(ddof=1)

    err = confidence_interval_mean(std, len(data), confidence)

    return mean, err



def estimate_std(data, confidence=0.95):
    """ estimate the standard deviation and the associated error of a
    data set """
    std = data.std()
    err = confidence_interval_std(std, len(data), confidence)

    return std, err



def estimate_cv(data, confidence=0.95):
    """ estimate the coefficient of variation and the associated error """
    mean = data.mean()
    std = data.std()
    cv = std/mean

    err = confidence_interval(cv, sampling_distribution_cv, (len(data),))

    return cv, err



if __name__ == '__main__':
    print("This file is intended to be used as a module only.")
