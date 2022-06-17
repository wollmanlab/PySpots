import os
import pandas as pd
import shutil
import dill as pickle
import tifffile
from MERFISH_Objects.FISHData import *
from datetime import datetime

class Utilities_Class(object):
    def __init__(self,
                 utilities_path):
        self.utilities_path = utilities_path
        if not os.path.exists(self.utilities_path):
            os.mkdir(self.utilities_path)
        
    def save_data(self,
                  data,
                  Dataset = 'Dataset',
                  Position = 'Position',
                  Hybe = 'Hybe',
                  Channel ='Channel',
                  Zindex = 'Zindex',
                  Type = 'Type'):
        if Type in ['image','stack']:
            File_Name = str(Dataset)+'_'+str(Position)+'_'+str(Hybe)+'_'+str(Channel)+'_'+str(Zindex)+'_'+str(Type)+'.tif'
            tifffile.imwrite(os.path.join(self.utilities_path,File_Name),data=data)
        elif Type in ['flag','log']:
            File_Name = str(Dataset)+'_'+str(Position)+'_'+str(Hybe)+'_'+str(Channel)+'_'+str(Zindex)+'_'+str(Type)+'.csv'
            fname = os.path.join(self.utilities_path,File_Name)
            with open(fname,"w+") as f:
                f.write(str(data))
                f.close()
        else:
            File_Name = str(Dataset)+'_'+str(Position)+'_'+str(Hybe)+'_'+str(Channel)+'_'+str(Zindex)+'_'+str(Type)+'.pkl'
            pickle.dump(data,open(os.path.join(self.utilities_path,File_Name),'wb'))
        try:
            os.chmod(os.path.join(self.utilities_path,File_Name), 0o777)
        except:
            do = 'nothing'
            # print('unable to set permissions: ',os.path.join(self.utilities_path,File_Name))
        
        
    def load_data(self,
                  Dataset = 'Dataset',
                  Position = 'Position',
                  Hybe = 'Hybe',
                  Channel ='Channel',
                  Zindex = 'Zindex',
                  Type = 'Type',
                  filename_only=False):
        try:
            if Type in ['image','stack']:
                File_Name = str(Dataset)+'_'+str(Position)+'_'+str(Hybe)+'_'+str(Channel)+'_'+str(Zindex)+'_'+str(Type)+'.tif'
                fname = os.path.join(self.utilities_path,File_Name)
                if filename_only:
                    data = fname
                else:
                    data = tifffile.imread(fname)
            elif Type in ['flag','log']:
                File_Name = str(Dataset)+'_'+str(Position)+'_'+str(Hybe)+'_'+str(Channel)+'_'+str(Zindex)+'_'+str(Type)+'.csv'
                fname = os.path.join(self.utilities_path,File_Name)
                if filename_only:
                    data = fname
                else:
                    with open(fname,"r") as f:
                        data = f.read()
                        f.close()
            else:
                File_Name = str(Dataset)+'_'+str(Position)+'_'+str(Hybe)+'_'+str(Channel)+'_'+str(Zindex)+'_'+str(Type)+'.pkl'
                fname = os.path.join(self.utilities_path,File_Name)
                if filename_only:
                    data = fname
                else:
                    data = pickle.load(open(fname,'rb'))
            return data
        except:
            return None
            
        
    def delete_data(self,
                  Dataset = 'Dataset',
                  Position = 'Position',
                  Hybe = 'Hybe',
                  Channel ='Channel',
                  Zindex = 'Zindex',
                  Type = 'Type'):
        File_Name = str(Dataset)+'_'+str(Position)+'_'+str(Hybe)+'_'+str(Channel)+'_'+str(Zindex)+'_'+str(Type)+'.pkl'
        os.remove(os.path.join(self.utilities_path,File_Name))
        
    def clear_data(self):
        shutil.rmtree(self.utilities_path)
        