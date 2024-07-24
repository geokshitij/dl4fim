###############################################################################
# Copyright (C) 2020, Owen Smith, University of North Georgia
# License: GPL v3.0
###############################################################################

import os
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from osgeo import gdal
from joblib import Parallel, delayed


class MWin:
    '''
    Initialize moving window

    Parameters
    ----------

    window_size : int, default=3
        Single integer size which indicates the i,j size of the moving
        window. E.g. window_size of 3 is equal to a window size of 3,3.
        Values must be odd numbers due to the way the window size is
        computed where the window_size // 2 is equal to the number of cells
        on each side of a cell is needed for a window.

    n_jobs : int, default=1
        Number of jobs to run in parallel across the CPU. -1 will use all
        available cores.

    Attributes
    ----------

    matrix : ndarray
        Output similarity matrix reshaped to initial i, j.


    '''
    def __init__(self, window_size=3, n_jobs=1):

        # Initialize
        self.__w = window_size
        self.threads = n_jobs
        self.__split_itr = []
        self.vector = []
        # Get d value which indicates the number of cells to move from the
        # original cell for the window size w.
        self.__d = window_size // 2
        self.verbosity = 0
        self.time = []

    def __verbose(func):
        # Decorative function for verbose time outputs
        def wrapper(self, *args, **kwargs):
            if self.verbosity == 1:
                start_time = time.time()
                func(self, *args, **kwargs)
                end_time = time.time() - start_time
                self.time.append(end_time)
                print(f"{end_time} Seconds")
            else:
                func(self, *args, **kwargs)
        return wrapper


    def __make_clusters(self, shape):

        if self.__rem !=0:
            self.__data_clusters = [[i, i + self.split_range] for i in
                    range(0, shape[0] - self.split_range,
                        self.split_range)]
            self.__data_clusters[-1][1] += self.__rem
        else:
            self.__data_clusters = [[i, i + self.split_range] for i in
                    range(0, shape[0], self.split_range)]


    def __neighbors(self, arr, i, j):
        '''
        Computes neighbor vector for window size w with consideration for outer
        edges.

        Parameters
        ----------

        arr : ndarray
            Input matrix to get cell i,j neighbors

        i, j : int
            Cell location i, j

        Returns
        -------

        n : ndarray
            Cell neighborhood vector
        '''

        # Use min and max to account for upper x and y neighbors
        # ranges of cells within the 'd' from the edges
        n = arr[max(i-self.__d,0):min(i+self.__d + 1,arr.shape[0]),
                max(j-self.__d,0):min(j+self.__d + 1,arr.shape[1])].flatten()
        return n

    def __mw(self, arr1, arr2, ii, j):
        '''
        Computes moving window algorithm as described by Costanza 1989.

        Parameters
        ----------

        arr1 : ndarray
            First array for comparison
        arr2 : ndarry
            Second array for comparison
        ii : int
            Cell location i
        j : int
            Cell location j

        Returns
        -------

        similarity : float
            Similarity value for cell i, j with neighborhood of window size
            'w'. Value will range between 0.0 - 1.0.
        '''

        # number of total cells across two neighborhoods with window size w.
        # Assign nodata value of -1 which will be ignored in final coefficient
        if arr1[ii][j] and arr2[ii][j] == self.nodata:
            return -1
        else:
            # Get neighborhood vectors of each cell i, j in arrays 1 and 2.
            a = self.__neighbors(arr1, ii, j)
            b = self.__neighbors(arr2, ii, j)
            # Find number of cells which are different values. A value of
            # 0 indicates it is the same.
            #d = sum(self.__compare_uni(p, a, b))
            a1_u, a1_c = np.unique(a, return_counts=True)
            a2_u, a2_c = np.unique(b, return_counts=True)
            a1 = dict(zip(a1_u, a1_c))
            a2 = dict(zip(a2_u, a2_c))
            p = set(a1_u).union(set(a2_u))
            d = 0

            for i in p:
                if i not in a1:
                    d += abs(0 - a2.get(i))
                elif i not in a2:
                    d += abs(a1.get(i) - 0)
                else:
                    d += abs(a1.get(i) - a2.get(i))
            # Divide number of cells which are different by the total number of
            # cells in the two neighborhoods. If it is 100% similar assign cell
            # the value of 1.
            w = (2 * len(a))
            similarity = (1 - d / w) if d != 0 else 1
            return similarity

    def moving_window(self, arr1, arr2):
        '''
        Moving window implementation where the number of cores utilized is one.

        Parameters
        ----------

        arr1 : ndarray
            First array for comparison
        arr2 : ndarry
            Second array for comparison

        Returns
        -------

        vector : list
            List vector which contains similarity values of each cell

        '''
        vector = []
        for ii in range(self.arr1.shape[0]):
            for j in range(self.arr1.shape[1]):
                vector.append(self.__mw(arr1, arr2, ii, j))
        return vector

    def split_moving_window(self, arr1, arr2, sl):
        '''
        Moving window implementation where the number of cores utilized is
        more than one.

        Parameters
        ----------

        arr1 : ndarray
            First array for comparison
        arr2 : ndarry
            Second array for comparison
        sl : list
            Section of arr1 and arr2 which has assigned to CPU core.

        Returns
        -------

        vector : list
            List vector which contains similarity values of each cell

        '''
        i1, i2 = sl[1]
        vector = []
        for ii in range(i1, i2):
            for j in range(arr1.shape[1]):
                vector.append(self.__mw(arr1, arr2, ii, j))
        return vector

    @__verbose
    def fit(self, x, y, nodata=None):
        '''
        Computes moving window algorithm.

        Parameters
        ----------

        x : ndarray, str
            Must be ndarray or a valid raster file path.
        y : ndarray, str
            Must be ndarray st(or a valid raster file path.
        nodata : int, default=None
            Only needs to be set if x and y are ndarrays, otherwise the nodata
            value is read from the raster datasets.
        '''

        if type(x) and type(y) == str:
            path1 = gdal.Open(x)
            path2 = gdal.Open(y)
            arr1 = path1.GetRasterBand(1).ReadAsArray()
            arr2 = path2.GetRasterBand(1).ReadAsArray()
            if not nodata:
                self.nodata = path1.GetRasterBand(1).GetNoDataValue()
            else:
                self.nodata = nodata
        elif type(x) and type(y) == np.ndarray:
            arr1 = x
            arr2 = y
            if not nodata:
                self.nodata = -9999
            else:
                self.nodata = nodata

        if arr1.shape != arr2.shape:
            raise Exception("Shape of x and y is not the same")

        self.i, self.j = arr1.shape
        # Create initial list of starts and stops for the split

        self.split_range = self.i // self.threads
        self.__rem = self.i % self.threads
        self.__make_clusters(arr1.shape)

        if self.threads == 1:
            results = []
            results.append(self.moving_window(arr1, arr2))
        else:
            slice = self.__data_clusters
            slice_dict = {}
            for i in range(len(slice)):
                slice_dict.update({i: slice[i]})
            results = Parallel(n_jobs=self.threads)(delayed(
                      self.split_moving_window)(arr1, arr2, sl)
                      for sl in slice_dict.items())

        vector = []
        for i in range(len(results)):
            vector += results[i]
        vector_arr = np.array(vector)

        tw = sum(1 for n in vector if n != -1)
        self.sim = np.sum(vector_arr) / (tw)

        mat = vector_arr.reshape([self.i, self.j])
        self.matrix = np.ma.masked_where(mat == -1,
                                    mat,
                                    copy=True)

        return results

    def plot(self, cmap="Greys"):

        plt.imshow(self.matrix, cmap=cmap)
        plt.show()

    def save_tif(self, snap, out_tif):
        if os.path.exists(out_tif):
            os.remove(out_tif)

        snap = gdal.Open(snap)

        driver = gdal.GetDriverByName('GTiff')
        metadata = driver.GetMetadata()
        shape = self.matrix.shape
        dst_ds = driver.Create(out_tif,
                               xsize=shape[1],
                               ysize=shape[0],
                               bands=1,
                               eType=gdal.GDT_Float32)
        proj = snap.GetProjection()
        geo = snap.GetGeoTransform()
        dst_ds.SetGeoTransform(geo)
        dst_ds.SetProjection(proj)
        dst_ds.GetRasterBand(1).SetNoDataValue(-1)
        dst_ds.GetRasterBand(1).WriteArray(self.matrix)
        dst_ds.FlushCache()
        dst_ds = None


class MultiResolution:

    def __init__(self, window_list, k=0, n_jobs=1):

        self.window_list = window_list
        self.weight = k
        self.threads = n_jobs
        self.sim = {}

    def fit(self, x, y, nodata=None):

        for i in range(len(self.window_list)):
             mw = MWin(self.window_list[i], self.threads)
             mw.fit(x, y, nodata)
             self.sim.update({self.window_list[i] : mw.sim})

        num = 0
        den = 0
        k = self.weight
        for i in range(len(self.window_list)):
            num += self.sim.get(self.window_list[i]) * math.e**(-k * self.window_list[i] - 1)
            den += math.e**(-k * self.window_list[i] - 1)

        self.ft = num / den

    def plot(self):

        dictonary_list = sorted(self.sim.items())
        x, y = zip(*dictonary_list)
        self.x = x
        self.y = y

        print(x)
        print(y)

        plt.plot(x, y)
        plt.show()

