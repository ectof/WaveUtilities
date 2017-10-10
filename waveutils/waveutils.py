import glob, os
import re
import pandas as pd
import numpy as np
import os.path as path
import xarray as xarray
import warnings
from xarray import Dataset as Dataset
from .helpers import Instruments


class WvSet(Dataset):

    def __init__(self,*args):
        super(WvSet, self).__init__(*args)
                
    
    def combine_first(self, new_ds):
        
        combined_ds = super(WvSet, self).combine_first(new_ds)
        combined_ds = WvSet(combined_ds)
        try:
            combined_ds.attrs = {"name": self.name + ",\," + new_ds.name,
            "dimension":self.dimension}
        except AttributeError:
            pass
        return combined_ds


def return_wave_path(wv, folder = "\\DataExport\\"):
    
    """Return the path to the wave.
    Typically not a user function.

    Args:
        wave: A wave number or list of wave number
        folder (optional): The folder to look for waves

    Returns:
        A list of wave paths.

    """
    currentDir = os.getcwd()
    fdrPath = folder
    # check if it's a list of wave numbers
    try:
        dataPath = [None] * len(wv)
    except:
        wv = list(wv)
        dataPath = [None] * len(wv)
    
    for i,v in enumerate(wv):
        # search for wv.txt files
        dataPath[i] = glob.glob(fdrPath + "*" + "_%d.txt" % v)
        
    return dataPath

def return_instruments(paths):

    """Return paths to specific instruments and how they
    should be scaled. Typically not a user function.

    Args:
        paths: A list of paths to wave files 

    Returns:
        A dictionary of instruments and paths

    """
    new_instruments = dict()
    
    for i,v in enumerate(paths):
        
        filename = path.split(v)[-1]
        tmp = re.match(r"([a-z]+)([0-9]+)", filename, re.I)
        tmp = tmp.group()
        new_instruments.update({tmp:v})
            
    return new_instruments

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

def load_wave(wv, folder = "/DataExport/", instruments = None,
    dims = ["major","minor"],
    majorscale = [1.,0], minorscale = [1.,0],
    fancy_dims = None):

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

    Returns:
        A list of xarray datasets each containing dataarrays with the variables

    """
    wvPath = return_wave_path(wv, folder = folder)
    wvList = [None] * len(wvPath)

    if isinstance(wv,list):
        pass
    else:
        wv = list(wv)

    for i,v in enumerate(wvPath):
        if not v:
            del wvPath[i]
            del wvList[i]
            warnings.warn("Index %d not found, dropping" % wv[i])
            del wv[i]

    for i,v in enumerate(wvPath):

        inst = return_instruments(v)
        
        for j,u in enumerate(inst.keys()):

            if j == 0:
                dimension = get_dimension(inst[u])

            if dimension != len(dims):
                dims = ["major","minor"]
                warnings.warn("Incorrect number of dimensions, resetting to [major,minor]")

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
                wvList[i] = xarray.Dataset({u:df})

            else:
                wvList[i].update({u:df})

        if instruments is not None:
            wvList[i] = instruments.output_gen(wvList[i].copy())

        wvList[i] = WvSet(wvList[i])
        
        if dimension == 1:
        
            wvList[i][dims[0]] = np.polyval(majorscale,wvList[i][dims[0]])
            wvList[i].sortby(wvList[i].coords[dims[0]],ascending=True)
            if fancy_dims is not None:
                try:
                    wvList[i][dims[0]].attrs = {"name":fancy_dims[dims[0]][0],
                    "units":fancy_dims[dims[0]][1]}
                except KeyError:
                    wvList[i][dims[0]].attrs = {"name":"major",
                    "units":""}

        elif dimension == 2:
        
            wvList[i][dims[0]] = np.polyval(majorscale,wvList[i][dims[0]])
            wvList[i][dims[1]] = np.polyval(minorscale,wvList[i][dims[1]])
            wvList[i] = wvList[i].sortby(wvList[i].coords[dims[0]],ascending=True)
            wvList[i] = wvList[i].sortby(wvList[i].coords[dims[1]],ascending=True)
            if fancy_dims is not None:
                try:
                    wvList[i][dims[0]].attrs = {"name":fancy_dims[dims[0]][0],
                    "units":fancy_dims[dims[0]][1]}
                    wvList[i][dims[1]].attrs = {"name":fancy_dims[dims[1]][0],
                    "units":fancy_dims[dims[1]][1]}
                except KeyError:
                    wvList[i][dims[0]].attrs = {"name":"major",
                    "units":""}
                    wvList[i][dims[1]].attrs = {"name":"minor",
                    "units":""}

        wvList[i].attrs = {"name":"%d" % wv[i],"dimension":dimension}
                
    return wvList




    
