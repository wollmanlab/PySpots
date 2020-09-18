import os
import pandas as pd
import shutil
import dill as pickle
import tifffile
from MERFISH_Objects.FISHData import *

class Utilities_Class(object):
    def __init__(self,
                 utilities_path):
#                  metadata_fname='metadata.pkl',
#                  columns=['Dataset','Position','Hybe','Channel','Zindex','Type','File_Name']):
        self.utilities_path = utilities_path
#         self.metadata_fname = metadata_fname
#         self.columns = columns
#         self.metadata_path = os.path.join(utilities_path,self.metadata_fname)
        if not os.path.exists(self.utilities_path):
            os.mkdir(self.utilities_path)
#         self.load_metadata()
        
    def save_data(self,
                  data,
                  Dataset = 'Dataset',
                  Position = 'Position',
                  Hybe = 'Hybe',
                  Channel ='Channel',
                  Zindex = 'Zindex',
                  Type = 'Type'):
#         self.metadata = self.metadata.append({'Dataset': Dataset,
#                                               'Position': Position,
#                                               'Hybe': Hybe,
#                                               'Channel':Channel,
#                                               'Zindex':str(Zindex),
#                                               'Type':Type,
#                                               'File_Name':File_Name},
#                                              ignore_index=True)
        if Type in ['image','stack']:
            File_Name = str(Dataset)+'_'+str(Position)+'_'+str(Hybe)+'_'+str(Channel)+'_'+str(Zindex)+'_'+str(Type)+'.tif'
            tifffile.imwrite(os.path.join(self.utilities_path,File_Name),data=data)
        else:
            File_Name = str(Dataset)+'_'+str(Position)+'_'+str(Hybe)+'_'+str(Channel)+'_'+str(Zindex)+'_'+str(Type)+'.pkl'
            pickle.dump(data,open(os.path.join(self.utilities_path,File_Name),'wb'))
        
        ### Stack needs axes swapped when saveing and loading
        
    def load_data(self,
                  Dataset = 'Dataset',
                  Position = 'Position',
                  Hybe = 'Hybe',
                  Channel ='Channel',
                  Zindex = 'Zindex',
                  Type = 'Type'):
#         File_Name = self.metadata[(self.metadata.Dataset==Dataset)&
#                       (self.metadata.Position==Position)&
#                       (self.metadata.Hybe==Hybe)&
#                       (self.metadata.Channel==Channel)&
#                       (self.metadata.Zindex==Zindex)&
#                       (self.metadata.Type==Type)]['File_Name']
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
            
#     def load_metadata(self):
#         if os.path.exists(self.metadata_path):
#             self.metadata = pickle.load(open(self.metadata_path,'rb'))
#         else:
#             self.metadata = pd.DataFrame(columns=self.columns)
        
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
        