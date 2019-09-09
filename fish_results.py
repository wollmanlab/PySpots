import dill as pickle
import os
import pandas
import numpy as np
from skimage.io import imread, imsave
from skimage.external import tifffile
import os
import pandas as pd
import numpy as np
import itertools as it

class SingleCellFishResults(object):
    def __init__(self, pth, ordered_gene_names):
        self.base_path = pth
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        self.genes = ordered_gene_names
        self.codeword_idx = np.arange(len(self.genes))
        self.expression_table = pd.DataFrame(columns = ['gene_vector'])
    @property
    def cell_ids(self):
        return self.expression_table.index
    @property
    def df(self):
        return self.expression_table
#     @property
#     def data(self, )
    def add_spot_counts(self, cell_id, counts, overwrite=False):
        if isinstance(counts, dict):
            counts = [counts[i] if i in counts else 0 for i in self.codeword_idx]
        if not isinstance(cell_id, int):
            raise ValueError("Cell ids must be integers")
        if cell_id in self.expression_table.index:
            if overwrite:
                print('Warning overwriting existing data.')
                self.expression_table.at[cell_id, 'gene_vector'] = counts
            else:
                raise ValueError('Cell id exists already and overwrite is false.')
        else:
            self.expression_table.at[cell_id, 'gene_vector'] = counts
    def add_data(self, cell_id, data_name, data, dtype=float):
        if not isinstance(cell_id, int):
            raise ValueError("Cell ids must be integers")
        if not cell_id in self.expression_table.index:
            raise ValueError("Ancillary data only supported for existing cells.")
        if not data_name in self.expression_table.columns:
            self.expression_table[data_name] = it.repeat(np.nan, self.expression_table.shape[0])
            
        self.expression_table.at[cell_id, data_name] = data
    def __getitem__(self, ix):
        if not ix in self.expression_table.index:
            raise ValueError("Cell id not in dataset.")
        else:
            return self.expression_table.loc[ix]
    def __iter__(self):
        self.n_i = 0
        return self
    def __next__(self):
        if self.n_i <= len(self.cellids):
            result = self.expression_table.loc[self.cell_ids[self.n_i]]
            self.n_i += 1
            return result
        else:
            raise StopIteration
        

        
        

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

    def regenerate_metadata(self,Return=False):
        # Generate a hybedata.csv from the existing files in the dir
        self.metadata = pandas.DataFrame(columns=['posname', 'zindex', 'dtype', 'filename'])
        dtypes = ['nf','cstk','cimg','mask','beads','tforms','spotcalls']
        for relative_fname in os.listdir(self.base_path):
            try:
                dtype = relative_fname.split('_')[0]
                zindex = relative_fname.split('_')[-1].split('.')[0]
                posname = relative_fname.split(dtype+'_')[1].split('_z')[0]
                if dtype in dtypes:
                    self.metadata = self.metadata.append({'posname': posname, 'zindex': str(zindex),
                                                          'dtype': dtype, 'filename': relative_fname},
                                                         ignore_index=True)
            except:
                if 'hybedata.csv' in relative_fname:
                    print(pos,' hybedata.csv')
                elif 'processing.pkl' in relative_fname:
                    print(pos,' processing.pkl')
                else:
                    print('Unexpexted File Found:')
                    print(pos,' ', relative_fname)
                continue
        self.metadata = self.metadata.drop_duplicates()
        self.metadata.to_csv(os.path.join(self.base_path, self.file_name),index=False)
        
        if Return == True:
            return self.metadata
        
    def load_metadata(self, pth):
        all_mds = []
        for subdir, curdir, filez in os.walk(pth):
            if self.file_name in filez:
                try:
                    temp = pandas.read_csv(os.path.join(self.base_path, subdir, self.file_name))
                    if len(temp) == 0:
                        # if the metadata is empty try and generate one from the file names
                        temp = self.regenerate_metadata(verbose=False)
                    temp.zindex=temp.zindex.astype(str)
                    all_mds.append(temp)
                except Exception as e:
                    print(self.file_name, e)
                    continue
        if len(all_mds)==0:
            try:
                self.metadata = self.regenerate_metadata(Return=True)
                return True
            except Exception as e:
                print(e)
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
            fname = "beads_{0}{1}{2}.pkl".format(posname, "_h_", zindex)
        elif dtype == 'tforms':
            fname = "tforms_{0}{1}{2}.pkl".format(posname, "_h_", zindex)
        elif dtype == 'spotcalls':
            fname = "spotcalls_{0}{1}{2}.pkl".format(posname, sep, zindex)
        return fname

    def add_and_save_data(self, data, posname, zindex, dtype, rewrite_metadata=True):
        relative_fname = self.generate_fname(posname, zindex, dtype)
        
        if relative_fname in self.metadata.filename:
            raise ValueError('Item exists in metadata already.')
        self.metadata = self.metadata.append({'posname': posname, 'zindex': str(zindex),
                                              'dtype': dtype, 'filename': relative_fname},
                                             ignore_index=True)
        full_fname = os.path.join(self.base_path, relative_fname)
        pth_part, filename_part = os.path.split(full_fname)
        if not os.path.exists(pth_part):
            os.makedirs(pth_part)
        save_passed = self.save_data(data, full_fname, dtype)
        if rewrite_metadata:
            self.metadata = self.metadata.drop_duplicates()
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
            pickle.dump(data,open(fname,'wb'))
        elif dtype == 'tforms':
            pickle.dump(data,open(fname,'wb'))
        elif dtype == 'spotcalls':
            pickle.dump(data,open(fname,'wb'))
        return True

    def get_data(self, posname, zindex, dtype, fname_only=False):
        if fname_only:
            try:
                dtype = fname_only.split('_')[0]
                zindex = fname_only.split('_')[-1].split('.')[0]
                posname = fname_only.split(dtype+'_')[1].split('_z')[0]
                out = self.load_data(posname, zindex, dtype)
                return out
            except:
                print('Couldnt generate "dtype" "zindex" or "posname" from filename provided')
                print('Try giving the filename without the full path')
    
        else:
            return self.load_data(posname, zindex, dtype)

    def lookup_data_filename(self, posname, zindex, dtype):
        subset = self.metadata[(self.metadata['posname']==posname) & (self.metadata['zindex']==str(zindex)) & (self.metadata['dtype']==dtype)]
        if subset.shape[0]==0:
#             print('filename failed to load generating new one')
            relative_fname = self.generate_fname(posname, zindex, dtype)
#             print(relative_fname)
            if os.path.isfile(os.path.join(self.base_path, relative_fname)):
                self.metadata = self.metadata.append({'posname': posname, 'zindex': str(zindex),
                                                      'dtype': dtype, 'filename': relative_fname},
                                                     ignore_index=True)
                return relative_fname
            else:
                return None
            return None
        else:
            return subset.filename.values[0]

    def load_data(self, posname, zindex, dtype):
        relative_fname = self.lookup_data_filename(posname, str(zindex), dtype)
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
            dout = pickle.load(open(full_fname,'rb'))
        elif dtype == 'tforms':
            dout = pickle.load(open(full_fname,'rb'))
        elif dtype == 'spotcalls':
            dout = pickle.load(open(full_fname,'rb'))
        return dout
