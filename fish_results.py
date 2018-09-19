import dill as pickle
import os
import pandas
import numpy as np
from skimage.io import imread, imsave
from skimage.external import tifffile


class HybeData(pickle.Pickler):
    def __init__(self, base_path, file_name='hybedata.csv'):
        #fpath = os.path.join(base_path, file_name)
        self.base_path = base_path
        self.file_name = file_name
        if os.path.exists(base_path):
            mdata_loaded = self.load_metadata(base_path)
        else:
            raise ValueError('Invalid initialization path provided. Does not exist.')
    def load_metadata(self, pth):
        all_mds = []
        for subdir, curdir, filez in os.walk(pth):
            if self.file_name in filez:
                try:
                    all_mds.append(pandas.read_csv(os.path.join(self.base_path, subdir, self.file_name)))
                except Exception as e:
                    print(e)
                    continue
        if len(all_mds)==0:
            self.metadata = pandas.DataFrame(columns=['posname', 'zindex', 'dtype', 'filename'])
            return False
        else:
            hdata = pandas.concat(all_mds, ignore_index=True)
            self.metadata = hdata
            return True
    def generate_fname(self, posname, zindex, dtype, sep="_z_"):
        if dtype == 'cstk':
            fname = "cstk_{0}{1}{2}.tif".format(posname, sep, zindex)
        elif dtype == 'nf':
            fname = "nf_{0}{1}{2}.csv".format(posname, sep, zindex)
        elif dtype == 'cimg':
            fname = "cimg_{0}{1}{2}.tif".format(posname, sep, zindex)
        relative_fname = os.path.join(fname)
        return relative_fname
    def add_and_save_data(self, data, posname, zindex, dtype, rewrite_metadata=True):
        relative_fname = self.generate_fname(posname, zindex, dtype)
        
        if relative_fname in self.metadata.filename:
            raise ValueError('Item exists in metadata already.')
        self.metadata = self.metadata.append({'posname': posname, 'zindex': zindex,
                                              'dtype': dtype, 'filename': relative_fname},
                                             ignore_index=True)
        full_fname = os.path.join(self.base_path, relative_fname)
        pth_part, filename_part = os.path.split(full_fname)
        if not os.path.exists(pth_part):
            os.makedirs(pth_part)
        save_passed = self.save_data(data, full_fname, dtype)
        if rewrite_metadata:
            self.metadata.to_csv(os.path.join(self.base_path, self.file_name))
    
    def save_data(self, data, fname, dtype):
        if fname is None:
            raise ValueError("No items for in metadata matching request.")
        if dtype == 'cstk':
            tifffile.imsave(fname, np.swapaxes(np.swapaxes(data,0,2),1,2), metadata={'axes': 'ZYX'})
        elif dtype == 'nf':
            dout = np.savetxt(fname, data)
        elif dtype == 'cimg':
            tifffile.imsave(fname, np.swapaxes(np.swapaxes(data,0,2),1,2), metadata={'axes': 'YX'})
        return True

    def get_data(self, posname, zindex, dtype, fname_only=False):
        if fname_only:
            return 'Not Implemented'
        else:
            return self.load_data(posname, zindex, dtype)
    def lookup_data_filename(self, posname, zindex, dtype):
        subset = self.metadata[(self.metadata.posname==posname) & (self.metadata.zindex==zindex) & (self.metadata.dtype==dtype)]
        if subset.shape[0]==0:
            return None
        else:
            return subset.filename.values[0]
    def load_data(self, posname, zindex, dtype):
        relative_fname = self.lookup_data_filename(posname, zindex, dtype)
        
        if relative_fname is None:
            raise ValueError("No items for in metadata matching request.")
        full_fname = os.path.join(self.base_path, relative_fname) 
        if dtype == 'cstk':
            dout = tifffile.imread(full_fname).astype(np.float64)
            dout = np.swapaxes(np.swapaxes(dout,0,2),0,1)
        elif dtype == 'nf':
            dout = np.genfromtxt(full_fname, delimiter=',')
        elif dtype == 'cimg':
            dout = tifffile.imread(full_fname).astype(np.int16)
            dout = np.swapaxes(np.swapaxes(dout,0,2),0,1)
        return dout
