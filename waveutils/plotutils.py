import glob, os
import re
import pandas as pd
import numpy as np
import string as sg
import pylab as pb
import warnings
from matplotlib.colors import LogNorm, SymLogNorm
import datetime

def niagara(data, indices = [None],
            delta = 0., cmap = "viridis",
            xlabel = "", ylabel = "", cblabel = "",
            putzeros = False):
    
    if indices == [None]:
        indices = data.index
    cs = [pb.cm.get_cmap(cmap)(i) for i in np.linspace(0,1.0,len(indices))]
    for i,v in enumerate(indices):
        img = pb.plot(data.iloc[i,:] + delta * i,color=cs[i])
        if putzeros:
            pb.plot(data.columns[[0,-1]],np.array([delta,delta])*i,'--k',lw=0.25)
    pb.xlabel(xlabel)
    pb.ylabel(ylabel)

    sm = pb.cm.ScalarMappable(cmap=pb.cm.get_cmap(cmap),
                               norm=pb.Normalize(vmin=indices[0],vmax=indices[-1]))
    sm._A = []
    pb.colorbar(sm,label = cblabel) 
    
    return img

def add_date(date_str="", location = "upper right", color = "k", fontsize = 6, box = True, **kwargs):

    location = location.split(" ")
    if location[0] == "lower":
        y = 0.
        va = "bottom"
    elif location[0] == "upper":
        y = 1.
        va = "top"
    else:
        warnings.warn("first location argument should be lower/upper, setting upper")
        y = 1.
        va = "top"

    if location[1] == "left":
        x = 0.0
        ha = "left"
    elif location[1] == "center":
        x = 0.5
        ha = "center"
    elif location[1] == "right":
        x = 1.
        ha = "right"
    else:
        warnings.warn("second location argument should be left/center/right, setting right")
        x = 1.
        ha = "right"

    bbox_props = dict(boxstyle="round", pad = 0.01, fc="w", ec="None", alpha=0.75)

    if date_str:
        pass
    else:
        date_str = datetime.datetime.now().strftime("%y/%m/%d-%H:%M")

    if box:
        pb.annotate(date_str, xy = [x,y], xycoords = "axes fraction",
            ha = ha, va = va, color = color, fontsize = fontsize, bbox = bbox_props)
    else:
        pb.annotate(date_str, xy = [x,y], xycoords = "axes fraction",
            ha = ha, va = va, color = color, fontsize = fontsize)

    return

def add_comment(comment, location = "lower left", color = "k", fontsize = 9, box = True, **kwargs):

    location = location.split(" ")
    if location[0] == "lower":
        y = 0.05
        va = "bottom"
    elif location[0] == "center":
        y = 0.5
        va = "center"
    elif location[0] == "upper":
        y = 0.95
        va = "top"
    else:
        warnings.warn("first location argument should be lower/center/upper, setting lower")
        y = 0.05
        va = "bottom"

    if location[1] == "left":
        x = 0.05
        ha = "left"
    elif location[1] == "center":
        x = 0.5
        ha = "center"
    elif location[1] == "right":
        x = 0.95
        ha = "right"
    else:
        warnings.warn("second location argument should be left/center/right, setting left")
        x = 0.05
        ha = "left"

    bbox_props = dict(boxstyle="round",pad = 0.2 , fc="w", ec="0.5", alpha=0.75)

    if box:
        pb.annotate(comment, xy = [x,y], xycoords = "axes fraction",
            ha = ha, va = va, color = color, fontsize = fontsize, bbox = bbox_props)
    else:
        pb.annotate(comment, xy = [x,y], xycoords = "axes fraction",
            ha = ha, va = va, color = color, fontsize = fontsize)

    return

def plot2d(dataframe, flip = False, colorbar = True, cblabel="",
    xlabel = "", ylabel = "", **kwargs):

    if flip:
        pb.pcolormesh(dataframe.columns,dataframe.index,dataframe,**kwargs)
    else:
        pb.pcolormesh(dataframe.index,dataframe.columns,dataframe.T,**kwargs)
    if colorbar:
        if cblabel:
            pb.colorbar(label = cblabel)
        else:
            pb.colorbar()

    if xlabel:
        pb.xlabel(xlabel)

    if ylabel:
        pb.ylabel(ylabel)

    return


def plot_multi(waves, item, titles = "", **kwargs):

    if not("vmax" in kwargs):
        vmax = waves[0].loc[item].get_values().max()
        for i in waves[1:]:
            vmax = max(vmax, i.loc[item].get_values().max())
        kwargs.update({"vmax":vmax})


    if not("vmin" in kwargs):
        vmin = waves[0].loc[item].get_values().min()
        for i in waves[1:]:
            vmin = min(vmin, i.loc[item].get_values().min())
        kwargs.update({"vmin":vmin})


    for i,v in enumerate(waves):
        if i < (len(waves) - 1):
            try:
                plot2d(v.loc[item],colorbar=False,**kwargs)
            except ValueError:
                ax = pb.gca()
                ax.remove()
                kwargs["vmin"] = kwargs["vmax"] * 1e-4
                plot2d(v.loc[item],colorbar=False,**kwargs)
                pass
        else:
            try:
                plot2d(v.loc[item],**kwargs)
            except ValueError:
                kwargs["vmin"] = kwargs["vmax"] * 1e-4
                plot2d(v.loc[item],**kwargs)
                pass

    return
    
