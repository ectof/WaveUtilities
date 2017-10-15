import glob, os
import re
import pandas as pd
import numpy as np
import string as sg
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import LogNorm, SymLogNorm
import datetime
import xarray

def niagara(dataArray, indices = None,
            delta = 0., cmap = "viridis",
            putzeros = False, nolabel = False,
           fancy_name = None):

    """Plot line cross sections across the second dimension of a xarray DataArray

    Args:
        dataArray: the data array
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
        indices = dataArray[dataArray.dims[0]]
    cs = [plt.cm.get_cmap(cmap)(i) for i in np.linspace(0,1.0,len(indices))]
    for i,v in enumerate(indices):
        img = plt.plot(dataArray[dataArray.dims[1]],
                      dataArray.loc[v,:] + delta * i,color=cs[i])
        if putzeros:
            plt.plot(dataArray[dataArray.dims[1]][[0,-1]],np.array([delta,delta])*i,'--k',lw=0.25)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap),
                               norm=plt.Normalize(vmin=indices[0],vmax=indices[-1]))
    sm._A = []
    
    ax_labels = list()
    for i in dataArray.dims:
        tmp = "\ (" + dataArray[i].attrs["units"] + ")$"
        tmp = "$" + dataArray[i].attrs["name"] + tmp
        ax_labels.append(tmp)
            
    if fancy_name is not None:
        if dataArray.name in fancy_name:
            ylabel = fancy_name[dataArray.name].pop()
    else:
        ylabel = "$%s" % dataArray.name
    if "units" in dataArray.attrs:
        unit_str = "\ (" + dataArray.attrs["units"] + ")$"
        ylabel = " ".join((ylabel,unit_str))

    if not(nolabel):
        plt.ylabel(ylabel)
        plt.xlabel(ax_labels[1])

    plt.colorbar(sm,label = ax_labels[0]) 

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

    y_dict = {"lower": [0.01, "bottom"], "center": [0.5, "center"],
    "upper": [0.99, "top"]}
    x_dict = {"left": [0.01, "left"], "center": [0.5, "center"],
    "right": [0.99, "right"]}

    location = location.split(" ")
    try:
        y = y_dict[location[0]]
    except KeyError:
        warnings.warn("first location argument should be lower/center/upper, setting lower")
        y = y_dict("lower")

    try:
        x = x_dict[location[1]]
    except KeyError:
        warnings.warn("second location argument should be left/center/right, setting right")
        x = x_dict("right")
        ha = "right"

    bbox_props = dict(boxstyle="round", pad = 0.01, fc="w", ec="None", alpha=0.75)

    if date_str:
        pass
    else:
        date_str = datetime.datetime.now().strftime("%y/%m/%d-%H:%M")

    if box:
        plt.annotate(date_str, xy = [x[0],y[0]], xycoords = "axes fraction",
            ha = x[1], va = y[1], color = color, fontsize = fontsize, bbox = bbox_props)
    else:
        plt.annotate(date_str, xy = [x,y], xycoords = "axes fraction",
            ha = x[1], va = y[1], color = color, fontsize = fontsize)

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
        if (i > 0) and (not((i+1)%new_line)) and (i<len(variables)-1):
            comment = comment + "$\n$"
    
    comment = comment + "$"
    return comment

def add_comment(comment, location = "upper right", color = "k",
    fontsize = 9, box = True, **kwargs):

    """Add a comment to a matplotlib plot

    Args:
        comment: the comment string
        location (optional): defaults to lower left
        color (optional): defaults to black
        fontsize (optional): defaults to 9
        box (boolean, optional): defau;ts to True


    """
    y_dict = {"lower": [0.05, "bottom"], "center": [0.5, "center"],
    "upper": [0.95, "top"]}
    x_dict = {"left": [0.05, "left"], "center": [0.5, "center"],
    "right": [0.95, "right"]}

    location = location.split(" ")
    try:
        y = y_dict[location[0]]
    except KeyError:
        warnings.warn("first location argument should be lower/center/upper, setting lower")
        y = y_dict("upper")

    try:
        x = x_dict[location[1]]
    except KeyError:
        warnings.warn("second location argument should be left/center/right, setting right")
        x = x_dict("right")

    bbox_props = dict(boxstyle="round",pad = 0.2 , fc="w", ec="0.5", alpha=0.75)

    if box:
        plt.annotate(comment, xy = [x[0],y[0]], xycoords = "axes fraction",
            ha = x[1], va = y[1], color = color, fontsize = fontsize, bbox = bbox_props)
    else:
        plt.annotate(comment, xy = [x[0],y[0]], xycoords = "axes fraction",
            ha = x[1], va = y[1], color = color, fontsize = fontsize)

    return


def plot_wv(dataArray,fancy_name = None, nolabel = False,
    add_colorbar = True, **kwargs):

    """Plot a single dataarray

    Args:
        dataArray: the data array
        indices (optional): where to plot the data as index values, defaults to None
            and plots all indices given
        add_colorbar (boolean, optional): add a colorbar to 2D plots, defaults to True
        nolabel (boolean, optional): don't put axis labels, defaults to False
        fancy_name (optional): A dict to look for the name of the instrument e.g.
            as a latex string. Not very useful, will be deprecated.

    Returns:
        The plot
    """

    dimension = len(dataArray.dims)
    
    ax_labels = [""] * 2
    for i,v in enumerate(dataArray.dims):
        try:
            tmp = "\ (" + dataArray[v].attrs["units"] + ")$"
            tmp = "$" + dataArray[v].attrs["name"] + tmp
            ax_labels[i] = tmp
        except:
            pass


    if dimension == 1:
        if fancy_name is not None:
            if dataArray.name in fancy_name:
                ylabel = fancy_name[dataArray.name].pop()
        else:
            ylabel = "$%s" % dataArray.attrs["name"]
        if "units" in dataArray.attrs:
            unit_str = "\ (" + dataArray.attrs["units"] + ")$"
            ylabel = " ".join((ylabel,unit_str))

        
        img = xarray.plot.line(dataArray,**kwargs)        
        if not(nolabel):
            plt.ylabel(ylabel)
            plt.xlabel(ax_labels[0])
        
    if dimension == 2:
        if fancy_name is not None:
            if dataArray.name in fancy_name:
                ylabel = fancy_name[dataArray.name].pop()
        else:
            ylabel = "$%s" % dataArray.attrs["name"]
        if "units" in dataArray.attrs:
            unit_str = "\ (" + dataArray.attrs["units"] + ")$"
            ylabel = " ".join((ylabel,unit_str))
        
        if add_colorbar:
            img = xarray.plot.pcolormesh(dataArray.T,
                               cbar_kwargs={"label":ylabel},
                               **kwargs)
        else:
            img = xarray.plot.pcolormesh(dataArray.T, add_colorbar = False,
                               **kwargs)
        
        if not(nolabel):
            plt.xlabel(ax_labels[0])
            plt.ylabel(ax_labels[1])

    return img


def plot_wvs(waves, item, put_title = True, title = None , **kwargs):

    """Plot waves

    Args:
        waves: list of xarray datasets
        item: the item to plot e.g. "g"
        put_title (optional): if true and the variablr "title" is not provided this defaults
        to the wave numbers
        title (optional): an optional title

    Returns:
        The plot
    """

    if not isinstance(waves,list):
        waves = [waves]

    if put_title and (title is None):
        make_title = True
        title = ""
    else:
        make_title = False

    if len(waves) > 1:
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

        if make_title:
            title += "\#" + v.name + "\ "
        
        if i == 0:
            if v.dimension == 2:
                ymin = v[item][v[item].dims[1]].min()
                ymax = v[item][v[item].dims[1]].max()
            xmin = v[item][v[item].dims[0]].min()
            xmax = v[item][v[item].dims[0]].max()
        else:
            if v.dimension == 2:
                ymin = min(ymin, v[item][v[item].dims[1]].min())
                ymax = max(ymax, v[item][v[item].dims[1]].max())
            xmin = min(xmin, v[item][v[item].dims[0]].min())
            xmax = max(xmax, v[item][v[item].dims[0]].max())        

        if i < (len(waves) - 1):
            try:
                img = plot_wv(v[item],add_colorbar=False,**kwargs)
            except ValueError:
                ax = plt.gca()
                ax.remove()
                try:
                    kwargs["vmin"] = kwargs["vmax"] * 1e-4
                except KeyError:
                    kwargs["vmin"] = 1e-4
                img = plot_wv(v[item],add_colorbar=False,**kwargs)

        else:
            try:
                img = plot_wv(v[item],**kwargs)
            except ValueError:
                ax = plt.gca()
                ax.remove()
                try:
                    kwargs["vmin"] = kwargs["vmax"] * 1e-4
                except KeyError:
                    kwargs["vmin"] = 1e-4
                img = plot_wv(v[item],**kwargs)

        if v.dimension == 2:
            plt.ylim(ymin,ymax)
        plt.xlim(xmin,xmax)
        if put_title:
            plt.title("$" + title + "$")

    return img


    
