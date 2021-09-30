import os
import pandas as pd
import shutil
import dill as pickle
import tifffile
from MERFISH_Objects.FISHData import *

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
        else:
            File_Name = str(Dataset)+'_'+str(Position)+'_'+str(Hybe)+'_'+str(Channel)+'_'+str(Zindex)+'_'+str(Type)+'.pkl'
            pickle.dump(data,open(os.path.join(self.utilities_path,File_Name),'wb'))
        
        
    def load_data(self,
                  Dataset = 'Dataset',
                  Position = 'Position',
                  Hybe = 'Hybe',
                  Channel ='Channel',
                  Zindex = 'Zindex',
                  Type = 'Type'):
        try:
            if Type in ['image','stack']:
                File_Name = str(Dataset)+'_'+str(Position)+'_'+str(Hybe)+'_'+str(Channel)+'_'+str(Zindex)+'_'+str(Type)+'.tif'
                data = tifffile.imread(os.path.join(self.utilities_path,File_Name))
            else:
                File_Name = str(Dataset)+'_'+str(Position)+'_'+str(Hybe)+'_'+str(Channel)+'_'+str(Zindex)+'_'+str(Type)+'.pkl'
                data = pickle.load(open(os.path.join(self.utilities_path,File_Name),'rb'))
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
        