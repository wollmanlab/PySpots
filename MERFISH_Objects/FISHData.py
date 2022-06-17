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
import anndata

class FISHData(object):
    def __init__(self, base_path, file_name='fishdata.csv'):
        self.base_path = base_path
        if not os.path.exists(base_path):
            os.mkdir(base_path)

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
        elif dtype == 'total_mask':
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
        elif dtype == 'h5ad':
            ftype = '.h5ad'
        else:
            ftype = '.npy'
        fname = "{0}_{1}_{2}_{3}_{4}_{5}{6}".format(dataset,posname,hybe,channel,zindex,dtype,ftype)
        return fname

    def add_and_save_data(self,data,dtype,dataset='X',posname='X',hybe='X',channel='X',zindex='X',rewrite_metadata=True):
        relative_fname = self.generate_fname(dataset,posname,hybe,channel,zindex,dtype)
        
        full_fname = os.path.join(self.base_path, relative_fname)
        pth_part, filename_part = os.path.split(full_fname)
        if not os.path.exists(pth_part):
            os.makedirs(pth_part)
        save_passed = self.save_data(data, full_fname, dtype)
    
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
        elif dtype == 'total_mask':
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
        elif dtype == 'h5ad':
            data.write(filename=fname)
        else:
            np.save(fname,data)
        try:
            os.chmod(fname, 0o777)
            return True
        except:
            return True



    def load_data(self, dtype,dataset='X',posname='X',hybe='X',channel='X',zindex='X'):
        relative_fname = self.generate_fname(dataset,posname,hybe,channel,zindex,dtype)
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
                elif dtype == 'total_mask':
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
                elif dtype == 'h5ad':
                    dout = anndata.read_h5ad(full_fname)
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