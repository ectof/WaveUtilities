import glob, os
import re
import pandas as pd
import numpy as np
import string as sg
import pylab as pb
import warnings
from matplotlib.colors import LogNorm, SymLogNorm
import datetime
import xarray

def niagara(dArray, indices = None,
            delta = 0., cmap = "viridis",
            putzeros = False, nolabel = False,
           fancy_name = None):

    """Plot line cross sections across the second dimension of a xarray DataArray

    Args:
        dArray: the data array
        indices (optional): where to plot the data as index values, defaults to None
            and plots all indices given
        delta (optional): amount to offset successive lines, defaults to 0
        cmap (optional): colormap to use, defaults to viridis
        putzeros (boolean, optional): draw lines at the zero of successive lines,
            defaults to False
        nolabel (boolean, optional): don't put axis labels, defaults to False
        fancy_name (optional): A dict to look for the name of the instrument e.g.
            as a latex string. Not very useful, will be deprecated.

    Returns:
        The plot
    """
    
    if indices is None:
        indices = dArray[dArray.dims[0]]
    cs = [pb.cm.get_cmap(cmap)(i) for i in np.linspace(0,1.0,len(indices))]
    for i,v in enumerate(indices):
        img = pb.plot(dArray[dArray.dims[1]],
                      dArray.loc[v,:] + delta * i,color=cs[i])
        if putzeros:
            pb.plot(dArray[dArray.dims[1]][[0,-1]],np.array([delta,delta])*i,'--k',lw=0.25)

    sm = pb.cm.ScalarMappable(cmap=pb.cm.get_cmap(cmap),
                               norm=pb.Normalize(vmin=indices[0],vmax=indices[-1]))
    sm._A = []
    
    ax_labels = list()
    for i in dArray.dims:
        tmp = "\ (" + dArray[i].attrs["units"] + ")$"
        tmp = "$" + dArray[i].attrs["name"] + tmp
        ax_labels.append(tmp)
            
    if fancy_name is not None:
        if dArray.name in fancy_name:
            ylabel = fancy_name[dArray.name].pop()
    else:
        ylabel = "$%s" % dArray.name
    if "units" in dArray.attrs:
        unit_str = "\ (" + dArray.attrs["units"] + ")$"
        ylabel = " ".join((ylabel,unit_str))

    if not(nolabel):
        pb.ylabel(ylabel)
        pb.xlabel(ax_labels[1])

    pb.colorbar(sm,label = ax_labels[0]) 

    return img

def add_date(date_str="", location = "upper right", color = "k",
    fontsize = 6, box = True, **kwargs):

    """Add a datestamp to a matplotlib plot

    Args:
        date_str (optional): defaults to the current date
        location (optional): defaults to upper right
        color (optional): defaults to black
        fontsize (optional): defaults to 6
        box (boolean, optional): defau;ts to True


    """

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

def add_comment(comment, location = "lower left", color = "k",
    fontsize = 9, box = True, **kwargs):

    """Add a comment to a matplotlib plot

    Args:
        comment: the comment string
        location (optional): defaults to lower left
        color (optional): defaults to black
        fontsize (optional): defaults to 9
        box (boolean, optional): defau;ts to True


    """

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


def plot_wv(dArray,fancy_name = None, nolabel = False,
    add_colorbar = True, **kwargs):

    """Plot a dataarray

    Args:
        dArray: the data array
        indices (optional): where to plot the data as index values, defaults to None
            and plots all indices given
        add_colorbar (boolean, optional): add a colorbar to 2D plots, defaults to True
        nolabel (boolean, optional): don't put axis labels, defaults to False
        fancy_name (optional): A dict to look for the name of the instrument e.g.
            as a latex string. Not very useful, will be deprecated.

    Returns:
        The plot
    """

    dimension = len(dArray.dims)
    
    ax_labels = [""] * 2
    for i,v in enumerate(dArray.dims):
        try:
            tmp = "\ (" + dArray[v].attrs["units"] + ")$"
            tmp = "$" + dArray[v].attrs["name"] + tmp
            ax_labels[i] = tmp
        except:
            pass


    if dimension == 1:
        if fancy_name is not None:
            if dArray.name in fancy_name:
                ylabel = fancy_name[dArray.name].pop()
        else:
            ylabel = "$%s" % dArray.name
        if "units" in dArray.attrs:
            unit_str = "\ (" + dArray.attrs["units"] + ")$"
            ylabel = " ".join((ylabel,unit_str))

        
        xarray.plot.line(dArray,**kwargs)        
        if not(nolabel):
            pb.ylabel(ylabel)
            pb.xlabel(ax_labels[0])
        
    if dimension == 2:
        if fancy_name is not None:
            if dArray.name in fancy_name:
                ylabel = fancy_name[dArray.name].pop()
        else:
            ylabel = "$%s" % dArray.name
        if "units" in dArray.attrs:
            unit_str = "\ (" + dArray.attrs["units"] + ")$"
            ylabel = " ".join((ylabel,unit_str))
        
        if add_colorbar:
            xarray.plot.pcolormesh(dArray.T,
                               cbar_kwargs={"label":ylabel},
                               **kwargs)
        else:
            xarray.plot.pcolormesh(dArray.T, add_colorbar = False,
                               **kwargs)
        
        if not(nolabel):
            pb.xlabel(ax_labels[0])
            pb.ylabel(ax_labels[1])

    return


def plot_multi(waves, item, titles = "", **kwargs):

    """Plot several waves

    Args:
        waves: list of xarray datasets
        item: the item to plot e.g. "g"
        titles (optional): doesn't do anything now

    Returns:
        The plot
    """

    if not("vmax" in kwargs):
        vmax = waves[0][item].max()
        for i in waves[1:]:
            vmax = max(vmax, i[item].max())
        kwargs.update({"vmax":vmax})


    if not("vmin" in kwargs):
        vmin = waves[0][item].min()
        for i in waves[1:]:
            vmin = min(vmin, i[item].min())
        kwargs.update({"vmin":vmin})


    for i,v in enumerate(waves):
        
        if i == 0:
            ymin = v[item][v[item].dims[1]].min()
            ymax = v[item][v[item].dims[1]].max()
            xmin = v[item][v[item].dims[0]].min()
            xmax = v[item][v[item].dims[0]].max()
        else:
            ymin = min(ymin, v[item][v[item].dims[1]].min())
            ymax = max(ymax, v[item][v[item].dims[1]].max())
            xmin = min(xmin, v[item][v[item].dims[0]].min())
            xmax = max(xmax, v[item][v[item].dims[0]].max())        

        if i < (len(waves) - 1):
            try:
                plot_wv(v[item],add_colorbar=False,**kwargs)
            except ValueError:
                ax = pb.gca()
                ax.remove()
                kwargs["vmin"] = kwargs["vmax"] * 1e-4
                plot_wv(v[item],add_colorbar=False,**kwargs)
                pass
        else:
            try:
                plot_wv(v[item],**kwargs)
            except ValueError:
                ax = pb.gca()
                ax.remove()
                kwargs["vmin"] = kwargs["vmax"] * 1e-4
                plot_wv(v[item],**kwargs)
                pass

        pb.ylim(ymin,ymax)
        pb.xlim(xmin,xmax)

    return

def make_comment(variables,fancy_dims=None,new_line=3):
    
    comment = "$"
    for i,key in enumerate(variables.keys()):
        if fancy_dims is not None:
            if key in fancy_dims:
                tmp = ("%s" % fancy_dims[key][0]) + ":\ "
                tmp = tmp + str(variables[key]) + "\ "
            else:
                tmp = ("%s" % key) + ":\ "
                tmp = tmp + str(variables[key]) + "\ "
        else:
            tmp = ("%s" % key) + ":\ "
            tmp = tmp + str(variables[key]) + "\ "
        
        comment = comment + tmp
        if (i > 0) and (not((i+1)%newline)) and (i<len(variables)-1):
            comment = comment + "$\n$"
    
    comment = comment + "$"
    return comment
    
