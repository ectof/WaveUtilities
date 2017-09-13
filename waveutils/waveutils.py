import glob, os
import re
import pandas as pd
import numpy as np
import os.path as path
import xarray as xarray
import warnings


# modifications to switch from panel to xarray

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
        wv = [wv]
        dataPath = [None] * len(wv)
    
    for i,v in enumerate(wv):
        # search for wv.txt files
        dataPath[i] = glob.glob(fdrPath + "*" + "_%d.txt" % v)
        
    return dataPath

def return_instruments(paths, instruments = None):

    """Return paths to specific instruments and how they
    should be scaled. Typically not a user function.

    Args:
        paths: A list of paths to wave files 
        instruments (optional): A pandas dataframe object with columns: name, scale and units.
            The index is the igor name of the variable, e.g. Vx1

    Returns:
        A pandas dataframe of instruments including the datafile path

    """

    if instruments is not None:
        try:
            if (instruments.keys() == ["name","scale","units"]).all():
                pass
            else:
                warnings.warn("Instruments should be a dataframe with \"name\" "
                              "e.g. \"Vx1\", \"scale\" e.g. 1e3, \"units\" e.g. \"V\" ")
        except:
            pass

    new_instruments = pd.DataFrame(columns = ["name","scale","units","path"])
    
    for i,v in enumerate(paths):
        
        filename = path.split(v)[-1]
        tmp = re.match(r"([a-z]+)([0-9]+)", filename, re.I)
        tmp = tmp.group()
        try:
            if (tmp in instruments.index):
                new_instruments.loc[tmp] = instruments.loc[tmp]
                new_instruments.loc[tmp,"path"] = v
            else:
                warnings.warn("%s found, but is not in instrument list" % tmp)
                new_instruments.loc[tmp] = [tmp,1.,"",v]
        except:
            new_instruments.loc[tmp] = [tmp,1.,"",v]
            
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
        instruments (optional): A pandas dataframe object with columns: name, scale and units.
            The index is the igor name of the variable, e.g. Vx1. Defaults to None
        dims (optional): names of the dimensions, defaults to major and minor
        majorscale (optional): list of polynomial coefficients through which the first
            dimension will be scaled, defaults to [1,0]
        minorscale (optional): list of polynomial coefficients through which the second
            dimension will be scaled, defaults to [1,0]
        fancy_dims (optional): a dictionary to look up fancy variable names for the dimensions.
            Keys should be name e.g. V_wire and units e.g. mV. These will be used for making plots

    Returns:
        A list of xarray datasets each containing dataarrays with the variables

    """

    wvPath = return_wave_path(wv, folder = folder)
    wvList = [None] * len(wvPath)

    for i,v in enumerate(wvPath):
        if not v:
            warnings.warn("Index %d not found, dropping" % wv[i])
            del wvPath[i]
            del wvList[i]
            del wv[i]

    for i,v in enumerate(wvPath):

        inst = return_instruments(v, instruments = instruments)
        
        for j,u in enumerate(inst.index):

            if j == 0:
                dimension = get_dimension(inst.loc[u,"path"])

            if dims is None:
                if dimension == 1:
                    dims = ["major"]
                else:
                    dims = ["major","minor"]

            if dimension == 1:
                df = pd.read_table(inst.loc[u,"path"], index_col=1,
                    skiprows=0, dtype=np.float64,
                    header=0, names=["crap", inst.loc[u,"name"]])
                df = df.dropna(axis=1,how="all")
                df = df.dropna()
                df = df.iloc[:,0]
                df.index.name = dims[0]
                df = df / inst.loc[u,"scale"]
                df = df.to_xarray()
                df.attrs = {"units":inst.loc[u,"units"]}


            elif dimension == 2:
                df = pd.read_table(inst.loc[u,"path"], index_col=1,
                    skiprows=2, dtype=np.float64, header=0)
                df = df.dropna(axis=1,how="all")
                df = df.dropna()
                df.columns = map(np.float64, df.columns)
                df = df / inst.loc[u,"scale"]
                df = xarray.DataArray(df.get_values(),
                    coords=[(dims[0],df.index),(dims[1],df.columns)])
                df.name = inst.loc[u,"name"]
                df.attrs = {"units":inst.loc[u,"units"]}


            if j == 0:
                wvList[i] = xarray.Dataset({inst.loc[u,"name"]:df})

            else:
                wvList[i].update(xarray.Dataset({inst.loc[u,"name"]:df}))

        
        if dimension == 1:
        
            wvList[i][dims[0]] = np.polyval(majorscale,wvList[i][dims[0]])
            wvList[i].sortby(wvList[i].coords[dims[0]],ascending=True)
            if fancy_dims is not None:
                    wvList[i][dims[0]].attrs = {"name":fancy_dims[dims[0]][0],
                                        "units":fancy_dims[dims[0]][1]}

        elif dimension == 2:
        
            wvList[i][dims[0]] = np.polyval(majorscale,wvList[i][dims[0]])
            wvList[i][dims[1]] = np.polyval(minorscale,wvList[i][dims[1]])
            wvList[i] = wvList[i].sortby(wvList[i].coords[dims[0]],ascending=True)
            wvList[i] = wvList[i].sortby(wvList[i].coords[dims[1]],ascending=True)
            if fancy_dims is not None:
                    wvList[i][dims[0]].attrs = {"name":fancy_dims[dims[0]][0],
                                        "units":fancy_dims[dims[0]][1]}
                    wvList[i][dims[1]].attrs = {"name":fancy_dims[dims[1]][0],
                                        "units":fancy_dims[dims[1]][1]}

        try:
            wvList[i].attrs = {"name":"%d" % wv[i]}
        except TypeError:
            wvList[i].attrs = {"name":"%d" % wv}

                
    return wvList




    
