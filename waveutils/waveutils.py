import glob, os
import re
import pandas as pd
import numpy as np
import string as sg
import os.path as path


def returnWavePath(wv,folder = "\\DataExport\\"):
    """ Return paths to the wave numbers, wv """
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
        ## print (fdrPath + "*" + "%d.txt" % v)
        dataPath[i] = glob.glob(fdrPath + "*" + "_%d.txt" % v)
        
    return dataPath

def thresholdmerge(smalldata, largedata, threshold):
    split = largedata.get_values() < threshold
    mergeddata = smalldata * split.astype(int) + largedata * (np.logical_not(split)).astype(int)
    
    return mergeddata

def getDimension(wvPath):
    
    fp = open(wvPath)
    fp.readline()
    line2 = fp.readline()
    
    if not(line2.split("\t")[1]):
        dimension = 2
    else:
        dimension = 1
    
    fp.close()

    return dimension

def loadWave(wv, folder = "/DataExport/", scale = None, name = None,
    majorscale = [1.,0], minorscale = [1.,0]):
    wvPath = returnWavePath(wv, folder = folder)
    wvFrame = [None] * len(wvPath)

    for i,v in enumerate(wvPath):
        wvFrame[i] = pd.DataFrame()
        list_ = []
        head = [None] * len(v)
        
        for j,u in enumerate(v):
            if (name != None) and (len(name) == len(head)):
                head = name        
            else:
                filename = path.split(u)[-1]
                head[j] = re.match(r"([a-z]+)([0-9]+)", filename, re.I)
                head[j] = "".join(head[j].groups())

            if j == 0:
                dimension = getDimension(u)

            if dimension == 1:
                df = pd.read_table(u, index_col=1, skiprows=0, dtype=np.float64,
                               header=0, names=["crap", head[j]])
                df = df.dropna(axis=1,how="all")
            elif dimension == 2:
                df = pd.read_table(u, index_col=1, skiprows=2, dtype=np.float64, header=0)
                df = df.dropna(axis=1,how="all")
                df = df.dropna()
                    
            if scale != None:
                try:
                    df = df / scale[j]
                    list_.append(df)
                except:
                    pass
            else:
                list_.append(df)
        
        if dimension == 1:
            wvFrame[i] = pd.concat(list_, axis=1)

            wvFrame[i].index = np.polyval(majorscale,wvFrame[i].index)
        
            if wvFrame[i].index[0] > wvFrame[i].index[-1]:
                wvFrame[i] = wvFrame[i].iloc[::-1,:]

        elif dimension ==2:
            wvFrame[i] = pd.Panel.from_dict(dict(zip(head,list_)))
            wvFrame[i].minor_axis = map(np.float64, wvFrame[i].minor_axis)

            wvFrame[i].major_axis = np.polyval(majorscale,wvFrame[i].index)
            wvFrame[i].minor_axis = np.polyval(minorscale,wvFrame[i].index)
        
            if wvFrame[i].minor_axis[0] > wvFrame[i].minor_axis[-1]:
                wvFrame[i] = wvFrame[i].iloc[:,:,::-1]
            if wvFrame[i].major_axis[0] > wvFrame[i].major_axis[-1]:
                wvFrame[i] = wvFrame[i].iloc[:,::-1,:]
                
    return wvFrame


    
