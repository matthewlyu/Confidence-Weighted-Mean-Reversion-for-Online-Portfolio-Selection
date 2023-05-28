import pandas as pd
import numpy as np
import sys
import os
import logging
import itertools
import multiprocessing
import contextlib
from cvxopt import solvers

solvers.options['show_progress'] = False


@contextlib.contextmanager
def mp_pool(n_jobs):
    n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    pool = multiprocessing.Pool(n_jobs)
    try:
        yield pool
    finally:
        pool.close()


def dataset(name):
    """ Return sample dataset from /data directory. """
    mod = sys.modules[__name__]
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', name + '.csv')
    data = pd.read_csv(filename)
    data = data.replace(0, np.nan).ffill()
    data = data.replace(0, np.nan).bfill()
    # Count the frequency of elements in the 'ticker' column
    value_counts = data['ticker'].value_counts()
    # Find the element with the highest frequency in the 'ticker' column, the stock code with the complete trading time
    max_freq = value_counts.max()
    most_common_elements = value_counts[value_counts == max_freq].index.tolist()
    # Construct a new dataframe based on the maximum frequency element,
    # with the column label as stock code and the column data as last price.
    last = pd.DataFrame()

    for element in most_common_elements:
        filtered_rows = data[data['ticker'] == element]
        last[element] = filtered_rows['last'].reset_index(drop=True)
    return last


def simplex_proj(y):
    """ Projection of y onto simplex. """
    m = len(y)
    bget = False

    s = sorted(y, reverse=True)
    tmpsum = 0.

    for ii in range(m - 1):
        tmpsum = tmpsum + s[ii]
        tmax = (tmpsum - 1) / (ii + 1)
        if tmax >= s[ii + 1]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + s[m - 1] - 1) / m

    return np.maximum(y - tmax, 0.)


def combinations(S, r):
    """ Generator of all r-element combinations of stocks from portfolio. """
    for ncols in itertools.combinations(S.columns, r):
        yield S[list(ncols)]


def log_progress(i, total, by=1):
    """ Log progress by pcts. """
    progress = ((100 * i / total) // by) * by
    last_progress = ((100 * (i - 1) / total) // by) * by

    if progress != last_progress:
        logging.debug('Progress: {}%...'.format(progress))


def freq(ix):
    """ Number of data items per minute. If data does not contain time index,
    assume 245 trading days per year."""
    assert isinstance(ix, pd.Index), 'freq method only accepts pd.Index object'

    # sort if data is not monotonic
    if not ix.is_monotonic:
        ix = ix.sort_values()

    if isinstance(ix, pd.DatetimeIndex):
        days = (ix[-1] - ix[0]).days
        return len(ix) / float(days) * 245.
    else:
        return 245.


# add alias to allow use of freq keyword in functions
_freq = freq
