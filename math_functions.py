'''
Created on Nov 21, 2014

@author: zwicker
'''

from __future__ import division

import fractions
import itertools
import math
import random
import sys

import numpy as np
from scipy import stats, misc


# make functions compatible with python 2 and 3
if sys.version_info[0] > 2:
    xrange = range  # @ReservedAssignment

# constants
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
    """ Returns an ordered sequence of `num` numbers from `start` to `end`,
    which are spaced logarithmically """

    if num is None:
        return np.logspace(np.log(start), np.log(end), base=np.e)
    else:
        return np.logspace(np.log(start), np.log(end), num, base=np.e)



def lognorm_mean_var_to_mu_sigma(mean, variance, definition='scipy'):
    """ determines the parameters of the log-normal distribution such that the
    distribution yields a given mean and variance. The optional parameter
    `definition` can be used to choose a definition of the resulting parameters
    that is suitable for the given software package. """
    mean2 = mean**2
    mu = mean2/np.sqrt(mean2 + variance)
    sigma = np.sqrt(np.log(1 + variance/mean2))
    if definition == 'scipy':
        return mu, sigma
    elif definition == 'numpy':
        return np.log(mu), sigma
    else:
        raise ValueError('Unknown definition `%s`' % definition)



def lognorm_mean(mean, sigma):
    """ returns a lognormal distribution parameterized by its mean and a spread
    parameter `sigma` """
    mu = mean * np.exp(-0.5 * sigma**2)
    return stats.lognorm(scale=mu, s=sigma)



def lognorm_mean_var(mean, variance):
    """ returns a lognormal distribution parameterized by its mean and its
    variance. """
    scale, sigma = lognorm_mean_var_to_mu_sigma(mean, variance, 'scipy')
    return stats.lognorm(scale=scale, s=sigma)



def random_log_uniform(v_min, v_max, size):
    """ returns random variables that a distributed uniformly in log space """
    log_min, log_max = np.log(v_min), np.log(v_max)
    res = np.random.uniform(log_min, log_max, size)
    return np.exp(res)



def _take_random_combinations_gen(data, r, num, repeat=False):
    """ a generator yielding `num` random combinations of length `r` of the 
    items in `data`. If `repeat` is False, none of the combinations is yielded
    twice. Note that the generator will be caught in a infinite loop if there
    are less then `num` possible combinations. """
    count, seen = 0, set()
    while True:
        # choose a combination
        s = tuple(sorted(random.sample(data, r)))
        # check whether it has been seen already
        if s in seen:
            continue
        # return the combination
        yield s
        # keep track of what combinations we have already seen
        if not repeat:
            seen.add(s)
        # check how many we have produced
        count += 1
        if count >= num:
            break
                
                
                
def take_combinations(iterable, r, num='all'):
    """ returns a generator yielding at most `num` random combinations of
    length `r` of the items in `iterable`. """
    if num == 'all':
        # yield all combinations
        return itertools.combinations(iterable, r)
    else:
        # check how many combinations there are
        data = list(iterable)
        num_combs = misc.comb(len(data), r, exact=True)
        if num_combs <= num:
            # yield all combinations
            return itertools.combinations(data, r)
        elif num_combs <= 10*num:
            # yield a chosen sample of the combinations
            choosen = set(random.sample(xrange(num_combs), num))
            gen = itertools.combinations(data, r)
            return (v for k, v in enumerate(gen) if k in choosen)
        else:
            # yield combinations at random
            return _take_random_combinations_gen(data, r, num)
        


def _take_random_product_gen(data, r, num, repeat=False):
    """ a generator yielding `num` random combinations of length `r` of the 
    items in `data`. If `repeat` is False, none of the combinations is yielded
    twice. Note that the generator will be caught in a infinite loop if there
    are less then `num` possible combinations. """
    raise NotImplementedError
    count, seen = 0, set()
    while True:
        # choose a combination
        s = tuple(sorted(random.sample(data, r)))
        # check whether it has been seen already
        if s in seen:
            continue
        # return the combination
        yield s
        # keep track of what combinations we have already seen
        if not repeat:
            seen.add(s)
        # check how many we have produced
        count += 1
        if count >= num:
            break
        
                
        
def take_product(data, r, num='all'):
    """ returns a generator yielding at most `num` random instances from the
    product set of `r` times the `data` """ 
    if num == 'all':
        # yield all combinations
        return itertools.product(data, repeat=r)
    else:
        # check how many combinations there are
        num_items = len(data)**r
        if num_items <= num:
            # yield all combinations
            return itertools.product(data, repeat=r)
        elif num_items <= 10*num:
            # yield a chosen sample of the combinations
            choosen = set(random.sample(xrange(num_items), num))
            gen = itertools.product(data, repeat=r)
            return (v for k, v in enumerate(gen) if k in choosen)
        else:
            # yield combinations at random
            return _take_random_product_gen(data, r, num)



def euler_phi(n):
    """ evaluates the Euler phi function for argument `n`
    See http://en.wikipedia.org/wiki/Euler%27s_totient_function
    Implementation based on http://stackoverflow.com/a/18114286/932593
    """
    amount = 0

    for k in xrange(1, n + 1):
        if fractions.gcd(n, k) == 1:
            amount += 1

    return amount



if __name__ == '__main__':
    print("This file is intended to be used as a module only.")
