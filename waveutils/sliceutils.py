import xarray as xarray
import numpy as np
from .plotutils_mpl import plot_wv


def get_slice(dataArray, direction = 0, plot = False, new_dim = "major"):

	""" Get a DataArray slice along a non-orthogonal axis

    Args:
        dataArray (optional): defaults to the current date
        location (optional): defaults to upper right
        color (optional): defaults to black
        fontsize (optional): defaults to 6
        box (boolean, optional): defau;ts to True


    """
	
	shape = np.shape(dataArray)
	index_min = min(range(len(shape)), key = shape.__getitem__)
	idc = np.empty((2,min(shape)),dtype=np.int64)

	if not direction:
		idc[index_min,:] = np.arange(shape[index_min],dtype=np.int64)
	else:
		idc[index_min,:] = np.flipud(np.arange(shape[index_min],dtype=np.int64))

	idc[int(not index_min),:] = np.linspace(0,max(shape)-1,num=shape[index_min],dtype=np.int64)
        
	data_slice = np.empty((3,min(shape)))
	data_slice[2,:] = [dataArray[i].values for i in zip(idc[0,:],idc[1,:])]
	data_slice[0,:] = [dataArray[dataArray.dims[0]][i].values for i in idc[0,:]]
	data_slice[1,:] = [dataArray[dataArray.dims[1]][i].values for i in idc[1,:]]

	if new_dim == "major":
		axes = range(2)
	elif new_dim == "minor":
		axes = range(1,-1,-1)
        
	dataArray_slice = xarray.DataArray(data_slice[2,:],
		coords = [data_slice[axes[0],:]],
		dims = [dataArray.dims[axes[0]]])
	dataArray_slice.attrs = dataArray.attrs
	dataArray_slice.name = dataArray.name
	dataArray_slice[dataArray_slice.dims[0]].attrs = dataArray[dataArray.dims[axes[0]]].attrs

	coord_slice = xarray.DataArray(data_slice[axes[1],:],
		coords = [data_slice[axes[0],:]],
		dims = [dataArray.dims[axes[0]]])
	coord_slice.attrs = dataArray[dataArray.dims[axes[1]]].attrs
	coord_slice.name = dataArray[dataArray.dims[axes[1]]].name
	coord_slice[dataArray_slice.dims[0]].attrs = dataArray[dataArray.dims[axes[0]]].attrs
    
	dataSet_slice = xarray.Dataset({dataArray.name: dataArray_slice,
			dataArray.dims[axes[1]]:coord_slice})
       
	if plot:
		plot_wv(dataSet_slice[dataArray.name])

	return dataSet_slice
