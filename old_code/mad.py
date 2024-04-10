"""
Calculate MAD values.
Version 0.2
author: Johannes Zschocke johannes.zschocke@uk-halle.de or
                          JohannesZschocke@t-online.de
modified by: Yaopeng Ma yaopeng.ma@biu.ac.il
                        yaopeng.ma97@gmail.com
date: 09/2022
"""

import numpy as np
import time
from numba import jit


def cal_mad(data, points):
    """
    Calculate MAD values. (by Johannes Zschocke)
    This routine calculates the MAD values based on the number of points
    (given by points) on the given 3D data.
    Parameters
    ----------
    data : 3D numpy array
        contains 3D data.
    points : integer
        number of datapoints to use for mad calculation.
    Returns
    -------
    mad : numpy array
        contains the calculated mad values.
    """
    # 'allocate' arrays and variables
    # the following variables are used as described and introduced by VahaYpya
    # 2015 DOI: 10.1111/cpf.12127

    # array for mad values
    mad = np.zeros(int(len(data[0]) / points))

    # array for r_i values
    r_i_array = np.empty(points)
    r_i_array[:] = np.nan

    # R_ave value
    R_ave = 0
    i_mad = 0

    # iterate over all values in data
    i = 0
    for (x, y, z) in zip(data[0], data[1], data[2]):
        r_i = np.sqrt(x**2 + y**2 + z**2)
        r_i_array[i] = r_i
        i += 1
        if (i == points):
            R_ave = np.nanmean(r_i_array)
            s = 0
            for ri in r_i_array:
                s += np.abs(ri - R_ave)

            s = s / points
            mad[i_mad] = s
            i_mad += 1
            r_i_array[:] = np.nan
            i = 0

    return mad


cal_mad_jit = jit(cal_mad)


if __name__ == "__main__":
    xyz = np.random.rand(3, 128 * 3600 * 8).astype(np.float32)
    start = time.time()
    mad = cal_mad(xyz, 128)
    print('pure python: ', time.time() - start)
    start = time.time()
    mad = cal_mad_jit(xyz, 128)
    start = time.time()
    mad2 = cal_mad_jit(xyz, 128)
    print('jit: ', time.time() - start)
    print(np.allclose(mad, mad2))

