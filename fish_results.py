import dill as pickle
import os
import pandas
import numpy as np
from skimage.io import imread, imsave
from skimage.external import tifffile
import os
import pandas as pd
import numpy as np

class SingleCellFishResults(object):
    def __init__(self, pth, ordered_gene_names):
        self.base_path = pth
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        self.genes = ordered_gene_names
        self.codeword_idx = np.arange(len(self.genes))
        self.expression_table = pd.DataFrame(columns = ['gene_vector'])
    def add_spot_counts(self, cell_id, counts, overwrite=False):
        if isinstance(counts, dict):
            counts = [counts[i] if i in counts else 0 for i in self.codeword_idx]
        if not isinstance(cell_id, int):
            raise ValueError("Cell ids must be integers")
        if cell_id in self.expression_table:
            if overwrite:
                print('Warning overwriting existing data.')
                self.expression_table.at[cell_id, 'gene_vector'] = counts
            else:
                raise ValueError('Cell id exists already and overwrite is false.')
        else:
            self.expression_table.at[cell_id, 'gene_vector'] = counts
    def add_data(self, cell_id, data_name, data):
        if not isinstance(cell_id, int):
            raise ValueError("Cell ids must be integers")
        if not cell_id in self.expression_table:
            raise ValueError("Ancillary data only supported for existing cells.")
        self.expression_table.at[cell_id, data_name] = data
        

class HybeData(pickle.Pickler):
    def __init__(self, base_path, file_name='hybedata.csv'):
        self.base_path = base_path
        self.file_name = file_name
        if self.base_path[-1]=='/':
            self.posname = os.path.split(self.base_path[:-1])[-1]
        else:
            self.posname = os.path.split(self.base_path)[-1]
        self.completed_hybes = {}
        if os.path.exists(base_path):
            mdata_loaded = self.load_metadata(base_path)
        else:
            self.metadata = pandas.DataFrame(columns=['posname', 'zindex', 'dtype', 'filename'])  # creates hybedata if it isn't there

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
        for i in self.metadata.columns:
            if 'Unnamed' in i:
                self.metadata = self.metadata.drop(columns=i)
            return True

    def generate_fname(self, posname, zindex, dtype, sep="_z_"):
        if dtype == 'cstk':
            fname = "cstk_{0}{1}{2}.tif".format(posname, sep, zindex)
        elif dtype == 'nf':
            fname = "nf_{0}{1}{2}.csv".format(posname, sep, zindex)
        elif dtype == 'cimg':
            fname = "cimg_{0}{1}{2}.tif".format(posname, sep, zindex)
        elif dtype == 'mask':
            fname = "mask_{0}{1}{2}.tif".format(posname, sep, zindex)
        elif dtype == 'beads':
            fname = "beads_{0}{1}{2}.csv".format(posname, "_h_", zindex)
        elif dtype == 'tforms':
            fname = "tforms.csv"
        elif dtype == 'spotcalls':
            fname = "spotcalls_{0}{1}{2}.pkl".format(posname, sep, zindex)
        return fname

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
            for i in self.metadata.columns:
            if 'Unnamed' in i:
                self.metadata = self.metadata.drop(columns=i)
            self.metadata.to_csv(os.path.join(self.base_path, self.file_name),index=False)

    def remove_metadata_by_zindex(self, zidx):
        self.metadata = self.metadata[self.metadata['zindex'] != zidx]
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        self.metadata.to_csv(os.path.join(self.base_path, self.file_name))
    
    def save_data(self, data, fname, dtype):
        if fname is None:
            raise ValueError("No items for in metadata matching request.")
        if dtype == 'cstk':
            data = data.astype('uint16')
            tifffile.imsave(fname, np.swapaxes(np.swapaxes(data,0,2),1,2), metadata={'axes': 'ZYX'})
        elif dtype == 'nf':
            dout = np.savetxt(fname, data)
        elif dtype == 'cimg':
            tifffile.imsave(fname, data.astype('int16')) # Allow imagej to read
        elif dtype == 'mask':
            tifffile.imsave(fname, data.astype('int16'))
        elif dtype == 'beads':
            data = data.drop_duplicates()
            dout = data.to_csv(fname,index=False)
        elif dtype == 'tforms':
            data = data.drop_duplicates()
            dout = data.to_csv(fname,index=False)
        elif dtype == 'spotcalls':
            pickle.dump(data,open(fname,'wb'))
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
            return None
        full_fname = os.path.join(self.base_path, relative_fname) 
        if dtype == 'cstk':
            dout = tifffile.imread(full_fname).astype(np.float64)
            dout = np.swapaxes(np.swapaxes(dout,0,2),0,1)
        elif dtype == 'nf':
            dout = np.genfromtxt(full_fname, delimiter=',')
        elif dtype == 'cimg':
            dout = tifffile.imread(full_fname).astype('int16')
        elif dtype == 'mask':
            dout = tifffile.imread(full_fname).astype('int16')
        elif dtype == 'beads':
            dout = pandas.read_csv(full_fname)
        elif dtype == 'tforms':
            dout = pandas.read_csv(full_fname)
        elif dtype == 'spotcalls':
            dout = pickle.load(open(full_fname,'rb'))
        return dout