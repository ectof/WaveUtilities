import numpy as np
import xarray as xarray
	
class Instruments(object):
    
    def __init__(self, outputs = None, functions = None, names = None, drop_inputs = True):

        self.outputs = outputs
        self.functions = functions
        self.names = names
        self.drop_inputs = drop_inputs

    def output_gen(self, data):
        
        outputs = self.outputs.keys()
        
        for i,v in enumerate(outputs):
            
            args = []
            for x in self.outputs[v]:
                try:
                    args.append(data[x])
                except KeyError:
                    args.append(x)
            new_da = self.functions[v](*args)
            new_da.attrs = {"name":self.names[v][0],"units":self.names[v][1]}
            data.update({v:new_da})
            
        if self.drop_inputs:
            for i,v in enumerate(data.var()):
                if v in self.outputs.keys():
                    pass
                else:
                    data = data.drop(v)
        
        return data