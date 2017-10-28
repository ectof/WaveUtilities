import glob, os
import re
import pandas as pd
import numpy as np
import os.path as path
import xarray as xarray
import warnings
from collections import OrderedDict
from xarray import Dataset as Dataset
from .helpers import Instruments


class WvSet(Dataset):

    def __init__(self,*args):
        super(WvSet, self).__init__(*args)
                
    
    def combine_first(self, new_ds):
        
        combined_ds = super(WvSet, self).combine_first(new_ds)
        combined_ds = WvSet(combined_ds)
        try:
            combined_ds.attrs = {"name": self.attrs["name"] + ",\," + new_ds.attrs["name"],
            "dimension":self.attrs["dimension"]}
        except AttributeError:
            pass
        return combined_ds


def return_wave_path(wv, folder = "\\DataExport\\", qcodes = False):
    
    """Return the path to the wave.
    Typically not a user function.

    Args:
        wave: A wave number or list of wave number
        folder (optional): The folder to look for waves

    Returns:
        A list of wave paths.

    """

    dataPath = [None] * len(wv)
    
    for i,v in enumerate(wv):
        # search for wv.txt files
        if qcodes:
            dataPath[i] = glob.glob(folder + "**/" + "#%03d_" % v + "*" + "/*.dat",
                recursive = True)
        else:
            dataPath[i] = glob.glob(folder + "*" + "_%d.txt" % v)
        
    return dataPath

def return_instruments_dimension(paths, qcodes = False):

    """Return paths to specific instruments and how they
    should be scaled. Typically not a user function.

    Args:
        paths: A list of paths to wave files 

    Returns:
        A dictionary of instruments and paths

    """
    new_instruments = OrderedDict()
    
    if qcodes:
        for i,v in enumerate(paths):
            fdr, filename = path.split(v)
            fp = open(v)
            line1 =fp.readline()
            fp.readline()
            line3 = fp.readline()
            fp.close()
            dimension = len(line3.split()) - 1
            inst = line1.split()[dimension+1:]
            for j,u in enumerate(inst):
                new_instruments.update({"_".join(u.split("_")[:-1]):v})
            
    else:
        for i,v in enumerate(paths):
            filename = path.split(v)[-1]
            tmp = re.match(r"([a-z]+)([0-9]+)", filename, re.I)
            tmp = tmp.group()
            new_instruments.update({tmp:v})
            if i == 0:
                dimension = get_dimension(v)
            
    return new_instruments, dimension

def thres_merge(smalldata, largedata, threshold):

    """Merge arrays based on a threshold. Where the data is larger than
    threshold the larger data will be used

    Args:
        smalldata: small data array
        largedata: large data array
        threshold: a number

    Returns:
        The merged array
    """

    split = np.abs(largedata) < threshold
    mergeddata = smalldata * split.astype(int) + largedata * (np.logical_not(split)).astype(int)
    
    return mergeddata

def get_dimension(wvPath):

    """Return the dimension of the wave at a given path

    Args:
        wvPath: the path to a wave

    Returns:
        The dimension i.e. 1 or 2.

    """
    
    try:
        fp = open(wvPath)
    except FileNotFoundError:
        raise SystemExit("Path to wave not found, exiting!")
    else:
        fp.readline()
        line2 = fp.readline()
    
    if not(line2.split("\t")[1]):
        dimension = 2
    else:
        dimension = 1
    
    fp.close()

    return dimension

def return_data_set(instruments, dimension, dims = ["major","minor"], qcodes = False):

    """Load the wave data into a xarray dataset

    Args:
        instruments: dict of instruments and paths 
        dimension: dimension (1 or 2)
        dims (optional): names of the dimensions, defaults to major and minor
        qcodes (optional): read data from qcodes format, defaults to False (i.e. Igor)

    Returns:
        A list of xarray datasets each containing dataarrays with the variables

    """
    
    if qcodes:
        
        filepath = next (iter (instruments.values()))
        fp = open(filepath)
        fp.readline()
        fp.readline()
        line3 = fp.readline()
        datashape = [int(i) for i in line3.split()[1:]]
        fp.close()
        data = np.loadtxt(filepath)
        if dimension == 2:
            minor_vals = data[:datashape[-1],1]
            major_vals = data[::datashape[-1],0]
        else:
            major_vals = data[:,0]

        
        for j,u in enumerate(instruments.keys()):
            if dimension == 2:
                df = xarray.DataArray(data[:,j+dimension].reshape(datashape[0], datashape[1]),
                    coords=[(dims[0],major_vals),(dims[1],minor_vals)])
            else:
                 df = xarray.DataArray(data[:,j+dimension],
                    coords=[(dims[0],datashape[0])])               
        
            if j == 0:
                data_set = xarray.Dataset({u:df})
            else:
                data_set.update({u:df})        
    else:
        
        for j,u in enumerate(instruments.keys()):

            if dimension == 1:
                df = pd.read_table(inst[u], index_col=1,
                    skiprows=0, dtype=np.float64,
                    header=0, names=["crap", u])
                df = df.dropna(axis=1,how="all")
                df = df.dropna()
                df = df.iloc[:,0]
                df.index.name = dims[0]
                df = df.to_xarray()

            elif dimension == 2:
                df = pd.read_table(inst[u], index_col=1,
                    skiprows=2, dtype=np.float64, header=0)
                df = df.dropna(axis=1,how="all")
                df = df.dropna()
                df.columns = map(np.float64, df.columns)
                df = xarray.DataArray(df.get_values(),
                    coords=[(dims[0],df.index),(dims[1],df.columns)])

            if j == 0:
                data_set = xarray.Dataset({u:df})
            else:
                data_set.update({u:df})
           
    return data_set

def load_wave(wv, folder = "/DataExport/", instruments = None,
    dims = ["major","minor"],
    majorscale = [1.,0], minorscale = [1.,0],
    fancy_dims = None, qcodes = False):

    """Load the wave data

    Args:
        wv: A wave index or list of indices
        folder (optional): the folder to look waves, defults to "DataExport"
        instruments (optional): An instacne of the instuments class
        dims (optional): names of the dimensions, defaults to major and minor
        majorscale (optional): list of polynomial coefficients through which the first
            dimension will be scaled, defaults to [1,0]
        minorscale (optional): list of polynomial coefficients through which the second
            dimension will be scaled, defaults to [1,0]
        fancy_dims (optional): dict to look up axis names and units for plotting
        qcodes (optional): read data from qcodes format, defaults to False (i.e. Igor)

    Returns:
        A list of xarray datasets each containing dataarrays with the variables

    """

    if isinstance(wv,list):
        pass
    else:
        try:
            wv = list(wv)
        except TypeError:
            wv = [wv]

    wvPath = return_wave_path(wv, folder = folder, qcodes = qcodes)
    wvList = [None] * len(wvPath)

    for i,v in enumerate(wvPath):
        if not v:
            del wvPath[i]
            del wvList[i]
            warnings.warn("Index %d not found, dropping" % wv[i])
            del wv[i]

    for i,v in enumerate(wvPath):

        inst, dimension = return_instruments_dimension(v, qcodes = qcodes)
        
        if dimension != len(dims):
            dims = ["major","minor"]
            warnings.warn("Incorrect number of dimensions, resetting to [major,minor]")

        wvList[i] = return_data_set(inst, dimension, dims = dims, qcodes = qcodes)

        if instruments is not None:
            wvList[i] = instruments.output_gen(wvList[i].copy())

        wvList[i] = WvSet(wvList[i])

        for j in range(dimension):
            wvList[i][dims[j]] = np.polyval(majorscale,wvList[i][dims[j]])
            wvList[i].sortby(wvList[i].coords[dims[j]],ascending=True)

        if fancy_dims is not None:
            for j in range(dimension):
                try:
                    wvList[i][dims[j]].attrs = {"name":fancy_dims[dims[j]][0],
                    "units":fancy_dims[dims[j]][1]}
                except KeyError:
                    wvList[i][dims[j]].attrs = {"name":"major",
                    "units":""}

        wvList[i].attrs = {"name":"%d" % wv[i],"dimension":dimension}
                
    return wvList




    
