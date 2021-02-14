import dill as pickle
import os
import pandas
import numpy as np
from skimage.io import imread, imsave
import cv2 #cv2.imread(os.path.join(fname),-1)
# from skimage.external import tifffile
import os
import pandas as pd
import numpy as np
import itertools as it

class FISHData(object):
    def __init__(self, base_path, file_name='fishdata.csv'):
        self.base_path = base_path
        if not os.path.exists(base_path):
            os.mkdir(base_path)
#         self.file_name = file_name
#         self.columns = ['dataset','posname','hybe','channel','zindex','dtype','filename']
#         self.dtypes = ['nf','cstk','cimg','mask','beads','tforms','spotcalls','flag','image']
#         if self.base_path[-1]=='/':
#             self.posname = os.path.split(self.base_path[:-1])[-1]
#         else:
#             self.posname = os.path.split(self.base_path)[-1]
#         self.completed_hybes = {}
#         if os.path.exists(base_path):
#             mdata_loaded = self.load_metadata(base_path)
#         else:
#             self.metadata = pandas.DataFrame(columns=self.columns)  # creates hybedata if it isn't there

#     def regenerate_metadata(self,Return=False,verbose=False):
#         # Generate a hybedata.csv from the existing files in the dir
#         self.metadata = pandas.DataFrame(columns=self.columns)
#         dtypes = ['nf','cstk','cimg','mask','beads','tforms','spotcalls','flag','image']
#         for relative_fname in os.listdir(self.base_path):
#             try:
#                 dataset = fname_only.split('_')[0]+'_'+fname_only.split('_')[1]
#                 posname = fname_only.split('_')[2]
#                 hybe = fname_only.split('_')[3]
#                 channel = fname_only.split('_')[4]
#                 zindex = fname_only.split('_')[5]
#                 dtype = fname_only.split('_')[6].split('.')[0]
#                 if dtype in dtypes:
#                     self.metadata = self.metadata.append({'dataset': dataset,
#                                                           'posname': posname,
#                                                           'hybe': hybe,
#                                                           'channel': channel,
#                                                           'zindex': str(zindex),
#                                                           'dtype': dtype,
#                                                           'filename': relative_fname},
#                                                          ignore_index=True)
#             except:
#                 if verbose:
#                     if 'hybedata.csv' in relative_fname:
#                         pass
#                     elif 'fishdata.csv' in relative_fname:
#                         pass
#                     elif 'processing.pkl' in relative_fname:
#                         pass
#                     else:
#                         print('Unexpexted File Found:')
#                         print(relative_fname)
#                 continue
#         self.metadata = self.metadata.drop_duplicates()
#         self.metadata.to_csv(os.path.join(self.base_path, self.file_name),index=False)
        
#         if Return == True:
#             return self.metadata
        
#     def load_metadata(self, pth):
#         all_mds = []
#         for subdir, curdir, filez in os.walk(pth):
#             if self.file_name in filez:
#                 try:
#                     temp = pandas.read_csv(os.path.join(self.base_path, subdir, self.file_name))
#                     if len(temp) == 0:
#                         # if the metadata is empty try and generate one from the file names
#                         temp = self.regenerate_metadata(verbose=False)
#                     temp.zindex=temp.zindex.astype(str)
#                     all_mds.append(temp)
#                 except Exception as e:
#                     print(self.file_name, e)
#                     continue
#         if len(all_mds)==0:
#             try:
#                 self.metadata = self.regenerate_metadata(Return=True)
#                 return True
#             except Exception as e:
#                 print(e)
#                 self.metadata = pandas.DataFrame(columns=self.columns)
#                 return False
#         else:
#             hdata = pandas.concat(all_mds, ignore_index=True)
#             self.metadata = hdata
#         for i in self.metadata.columns:
#             if 'Unnamed' in i:
#                 self.metadata = self.metadata.drop(columns=i)
#             return True

    def generate_fname(self,dataset,posname,hybe,channel,zindex,dtype):
        if dtype == 'cstk':
            ftype = '.tif'
        elif dtype =='image':
            ftype = '.tif'
        elif dtype == 'nf':
            ftype = '.csv'
        elif dtype == 'flag':
            ftype = '.csv'
        elif dtype == 'log':
            ftype = '.csv'
        elif dtype == 'cimg':
            ftype = '.tif'
        elif dtype == 'nuclei_mask':
            ftype = '.tif'
        elif dtype == 'cytoplasm_mask':
            ftype = '.tif'
        elif dtype == 'beads':
            ftype = '.csv'
        elif dtype == 'tforms':
            ftype = '.pkl'
        elif dtype == 'spotcalls':
            ftype = '.csv'
        elif dtype == 'cell_metadata':
            ftype = '.csv'
        elif dtype == 'counts':
            ftype = '.csv'
        else:
            ftype = '.npy'
        fname = "{0}_{1}_{2}_{3}_{4}_{5}{6}".format(dataset,posname,hybe,channel,zindex,dtype,ftype)
        return fname

    def add_and_save_data(self,data,dtype,dataset='X',posname='X',hybe='X',channel='X',zindex='X',rewrite_metadata=True):
        relative_fname = self.generate_fname(dataset,posname,hybe,channel,zindex,dtype)
        
#         if relative_fname in self.metadata.filename:
#             raise ValueError('Item exists in metadata already.')
#             # What about overwriting items
#         self.metadata = self.metadata.append({'dataset': dataset,
#                                               'posname': posname,
#                                               'hybe': hybe,
#                                               'channel': channel,
#                                               'zindex': str(zindex),
#                                               'dtype': dtype,
#                                               'filename': relative_fname},
#                                              ignore_index=True)
        full_fname = os.path.join(self.base_path, relative_fname)
        pth_part, filename_part = os.path.split(full_fname)
        if not os.path.exists(pth_part):
            os.makedirs(pth_part)
        save_passed = self.save_data(data, full_fname, dtype)
#         if rewrite_metadata:
#             self.metadata = self.metadata.drop_duplicates()
#             for i in self.metadata.columns:
#                 if 'Unnamed' in i:
#                     self.metadata = self.metadata.drop(columns=i)
#             self.metadata.to_csv(os.path.join(self.base_path, self.file_name),index=False)

#     def remove_metadata_by_zindex(self, zidx):
#         self.metadata = self.metadata[self.metadata['zindex'] != zidx]
#         if not os.path.exists(self.base_path):
#             os.makedirs(self.base_path)
#         self.metadata.to_csv(os.path.join(self.base_path, self.file_name))
    
    def save_data(self, data, fname, dtype):
        if fname is None:
            raise ValueError("No items for in metadata matching request.")
        if dtype == 'cstk':
            cv2.imwrite(fname, np.swapaxes(np.swapaxes(data.astype('uint16'),0,2),1,2))
#             tifffile.imsave(fname, np.swapaxes(np.swapaxes(data.astype('uint16'),0,2),1,2), metadata={'axes': 'ZYX'})
        elif dtype == 'nf':
            dout = np.savetxt(fname, data)
        elif dtype == 'flag':
            with open(fname,"w+") as f:
                f.write(str(data))
                f.close()
        elif dtype == 'log':
            with open(fname,"w+") as f:
                f.write(str(data))
                f.close()
        elif dtype == 'cimg':
            cv2.imwrite(fname, data.astype('uint16'))
#             tifffile.imsave(fname, data.astype('int16'))
        elif dtype == 'nuclei_mask':
            cv2.imwrite(fname, data.astype('uint16'))
#             tifffile.imsave(fname, data.astype('int16'))
        elif dtype == 'cytoplasm_mask':
            cv2.imwrite(fname, data.astype('uint16'))
#             tifffile.imsave(fname, data.astype('int16'))
        elif dtype == 'beads':
            pd.DataFrame(data).to_csv(fname)
        elif dtype == 'tforms':
            pickle.dump(data,open(fname,'wb'))
        elif dtype == 'spotcalls':
            data.to_csv(fname)
        elif dtype == 'cell_metadata':
            data.to_csv(fname)
        elif dtype == 'counts':
            data.to_csv(fname)
        elif dtype == 'image':
            cv2.imwrite(fname, data.astype('uint16'))
#             tifffile.imsave(fname, data.astype('uint16'))
        else:
            np.save(fname,data)
        return True

#     def get_data(self,dataset,posname,hybe,channel,zindex,dtype,fname_only=False):
#         if fname_only:
#             try:
#                 dataset = fname_only.split('_')[0]
#                 posname = fname_only.split('_')[1]
#                 hybe = fname_only.split('_')[2]
#                 channel = fname_only.split('_')[3]
#                 zindex = fname_only.split('_')[4]
#                 dtype = fname_only.split('_')[5].split('.')[0]
#                 out = self.load_data(posname, zindex, dtype)
#                 return out
#             except:
#                 print('Couldnt generate "dtype" "zindex" or "posname" from filename provided')
#                 print('Try giving the filename without the full path')
    
#         else:
#             return self.load_data(dataset,posname,hybe,channel,zindex,dtype)

#     def lookup_data_filename(self,dataset,posname,hybe,channel,zindex,dtype):
#         subset = self.metadata[(self.metadata['dataset']==dataset) &
#                                (self.metadata['posname']==posname) &
#                                (self.metadata['hybe']==hybe) &
#                                (self.metadata['channel']==channel) &
#                                (self.metadata['zindex']==str(zindex)) &
#                                (self.metadata['dtype']==dtype)]
#         if subset.shape[0]==0:
#             relative_fname = self.generate_fname(dataset,posname,hybe,channel,zindex,dtype)
#             if os.path.isfile(os.path.join(self.base_path, relative_fname)):
#                 self.metadata = self.metadata.append({'dataset': dataset,
#                                                       'posname': posname,
#                                                       'hybe': hybe,
#                                                       'channel': channel,
#                                                       'zindex': str(zindex),
#                                                       'dtype': dtype,
#                                                       'filename': relative_fname},
#                                                      ignore_index=True)
#                 return relative_fname
#             else:
#                 return None
#             return None
#         else:
#             return subset.filename.values[0]

    def load_data(self, dtype,dataset='X',posname='X',hybe='X',channel='X',zindex='X'):
        relative_fname = self.generate_fname(dataset,posname,hybe,channel,zindex,dtype)
#         relative_fname = self.lookup_data_filename(dataset,posname,hybe,channel,str(zindex),dtype)
#         if relative_fname is None:
#             return None
        full_fname = os.path.join(self.base_path, relative_fname)
        if os.path.exists(full_fname):
            try:
                if dtype == 'cstk':
                    dout = cv2.imread(full_fname,-1).astype(np.float64)
#                     dout = tifffile.imread(full_fname).astype(np.float64)
                    dout = np.swapaxes(np.swapaxes(dout,0,2),0,1)
                elif dtype == 'image':
                    dout = cv2.imread(full_fname,-1).astype(np.float64)
#                     dout = tifffile.imread(full_fname).astype(np.float64)
                elif dtype == 'nf':
                    dout = np.genfromtxt(full_fname, delimiter=',')
                elif dtype == 'flag':
                    with open(full_fname,"r") as f:
                        dout = f.read()
                        f.close()
                elif dtype == 'log':
                    with open(full_fname,"r") as f:
                        dout = f.read()
                        f.close()
                elif dtype == 'cimg':
                    dout = cv2.imread(full_fname,-1).astype('int16')
#                     dout = tifffile.imread(full_fname).astype('int16')
                elif dtype == 'nuclei_mask':
                    dout = cv2.imread(full_fname,-1).astype('int16')
#                     dout = tifffile.imread(full_fname).astype('int16')
                elif dtype == 'cytoplasm_mask':
                    dout = cv2.imread(full_fname,-1).astype('int16')
#                     dout = tifffile.imread(full_fname).astype('int16')
                elif dtype == 'beads':
                    dout = np.array(pd.read_csv(full_fname,index_col=0))
                elif dtype == 'tforms':
        #             dout = pd.read_csv(full_fname,index_col=0)
                    dout = pickle.load(open(full_fname,'rb'))
                elif dtype == 'spotcalls':
                    dout = pd.read_csv(full_fname,index_col=0)
                elif dtype == 'cell_metadata':
                    dout = pd.read_csv(full_fname,index_col=0)
                elif dtype == 'counts':
                    dout = pd.read_csv(full_fname,index_col=0)
                else:
                    try:
                        dout = np.load(full_fname)
                    except:
                        dout = None
            except Exception as e:
                print('unable to read file')
                print('Error:',e,full_fname)
                dout = None
        else:
            dout = None
        return dout