from tqdm import tqdm
from MERFISH_Objects.Position import *
from MERFISH_Objects.Classify import *
import pandas as pd
from metadata import Metadata
from hybescope_config.microscope_config import *
import numpy as np
import dill as pickle
import importlib
from tqdm import tqdm
import cv2
import random
from skimage.filters import threshold_otsu
from MERFISH_Objects.FISHData import *
import dill as pickle
from datetime import datetime

class Dataset_Class(object):
    def __init__(self,
                 metadata_path,
                 dataset,
                 cword_config,
                 verbose=True):
        """Class to Process Dataset
        Includes Finding Hot Pixels
        Includes Filtering out False Transcripts
        Includes Final Counts matrix 

        Args:
            metadata_path (str): path to raw data
            dataset (str): name of dataset
            cword_config (str): name of config module
            verbose (bool, optional): _description_. Defaults to True.
        """
        self.metadata_path = metadata_path
        self.dataset = dataset
        self.cword_config = cword_config
        self.verbose = verbose
        self.merfish_config = importlib.import_module(self.cword_config)
        self.parameters = self.merfish_config.parameters
        self.daemon_path = self.parameters['daemon_path']
        if not os.path.exists(self.daemon_path):
            os.mkdir(self.daemon_path)
        self.position_daemon_path = os.path.join(self.daemon_path,'position')
        if not os.path.exists(self.position_daemon_path):
            os.mkdir(self.position_daemon_path)
            os.mkdir(os.path.join(self.position_daemon_path,'input'))
            os.mkdir(os.path.join(self.position_daemon_path,'output'))
        self.bitmap = self.merfish_config.bitmap
        self.channels = list(np.unique([channel for seq,hybe,channel in self.bitmap]))
        self.hybes = list(np.unique([hybe for seq,hybe,channel in self.bitmap]))
        self.utilities = Utilities_Class(self.parameters['utilities_path'])
        self.fishdata = FISHData(os.path.join(self.metadata_path,self.parameters['fishdata']))
        
        self.projection_zstart=self.parameters['projection_zstart'] 
        self.projection_k=self.parameters['projection_k']
        self.projection_zskip=self.parameters['projection_zskip'] 
        self.projection_zend=self.parameters['projection_zend']
        self.projection_function=self.parameters['projection_function']
        self.two_dimensional = self.parameters['two_dimensional']
        
        
        self.n_pos = 10
        self.completed = False
        self.passed = True
        
    def run(self):
        self.main()
        
    def main(self):
        self.check_imaging()
        self.check_hot_pixel()
        self.check_flags()
    
    def update_user(self,message):
        """ For User Display"""
        i = [i for i in tqdm([],desc=str(datetime.now().strftime("%H:%M:%S"))+' '+str(message))]        
        
    def check_imaging(self):
        if self.verbose:
            self.update_user('Checking Imaging')
        # self.metadata = Metadata(self.metadata_path)
        self.acqs = [i for i in os.listdir(self.metadata_path) if 'hybe' in i]
        self.metadata = Metadata(os.path.join(self.metadata_path,self.acqs[0]))
        self.posnames = self.metadata.image_table[self.metadata.image_table.acq.isin(self.acqs)].Position.unique()
        
    def check_hot_pixel(self):
        if self.verbose:
            self.update_user('Checking Hot Pixel')
        self.hotpixel = self.utilities.load_data(Dataset=self.dataset,Type='hot_pixels')
        if isinstance(self.hotpixel,type(None)):
            self.find_hot_pixels(std_thresh=self.parameters['std_thresh'],
                                 n_acqs=self.parameters['n_acqs'],
                                 kernel_size=self.parameters['hotpixel_kernel_size'])
            
    def check_flags(self):
        self.utilities.save_data('Started',Dataset=self.dataset,Type='flag')
        if self.verbose:
            iterable = tqdm(self.posnames,desc=str(datetime.now().strftime("%H:%M:%S"))+' Checking Position Flags')
        else:
            iterable = self.posnames
        self.started = []
        self.passed = []
        self.not_started = []
        self.failed = []
        for posname in iterable:
            flag = self.utilities.load_data(Dataset=self.dataset,Position=posname,Type='flag')
            if isinstance(flag,type(None)):
                self.not_started.append(posname)
            elif flag == 'Started':
                fname = self.dataset+'_'+posname+'.pkl'
                fname_path = os.path.join(self.position_daemon_path,'input',fname)
                if os.path.exists(fname_path):
                    self.started.append(posname)
                else:
                    self.not_started.append(posname)
            elif flag == 'Passed':
                self.passed.append(posname)
            elif flag =='Failed':
                self.failed.append(posname)
        if len(self.acqs)>1: # All positions have been imaged atleast once
            if len(self.not_started)==0: # All positions have been started
                if len(self.started)==0: # All positions have been completed
                    self.process_transcripts()
            else:
                self.create_positions()
                
    def create_positions(self):
        if self.verbose:
            iterable = tqdm(self.not_started,desc=str(datetime.now().strftime("%H:%M:%S"))+' Creating Positions')
        else:
            iterable = self.not_started
        for posname in iterable:
            fname = self.dataset+'_'+posname+'.pkl'
            fname_path = os.path.join(self.position_daemon_path,'input',fname)
            data = {'metadata_path':self.metadata_path,
                    'dataset':self.dataset,
                    'posname':posname,
                    'cword_config':self.cword_config,
                    'level':'position'}
            pickle.dump(data,open(fname_path,'wb'))
            flag = self.utilities.save_data('Started',Dataset=self.dataset,Position=posname,Type='flag')
            
    def find_hot_pixels(self,std_thresh=3,n_acqs=5,kernel_size=3):
        if self.verbose:
            self.update_user('Loading Metadata')
        self.metadata = Metadata(self.metadata_path)
        if self.verbose:
            self.update_user('Finding Hot Pixels')
        if kernel_size%2==0:
            kernel_size = kernel_size+1
        kernel = np.ones((kernel_size,kernel_size))
        kernel[int(kernel_size/2),int(kernel_size/2)] = 0
        kernel = kernel/np.sum(kernel)
        X = []
        Y = []
        hot_pixel_dict = {}
        if len(self.posnames)>self.n_pos:
            pos_sample = random.sample(list(self.posnames),self.n_pos)
        else:
            pos_sample = self.posnames
        if self.verbose:
            iterable = tqdm(pos_sample,desc=str(datetime.now().strftime("%H:%M:%S"))+' Finding Hot Pixels')
        else:
            iterable = pos_sample
        for pos in iterable:
            pos_md =  self.metadata.image_table[self.metadata.image_table.Position==pos]
            acqs = pos_md.acq.unique()
            if len(acqs)>n_acqs:
                acqs = random.sample(list(acqs),n_acqs)
            for acq in acqs:
                hot_pixel_dict[acq] = {}
                channels = pos_md[pos_md.acq==acq].Channel.unique()
                channels = set(list(channels)).intersection(self.channels)
                for channel in channels:
                    img = np.average(self.metadata.stkread(Position=pos,Channel=channel,acq=acq),axis=2)
                    bkg_sub = img-cv2.filter2D(img,-1,kernel)
                    avg = np.average(bkg_sub)
                    std = np.std(bkg_sub)
                    thresh = (avg+(std_thresh*std))
                    loc = np.where(bkg_sub>thresh)
                    X.extend(loc[0])
                    Y.extend(loc[1])
        # Need to Doube Check
        img = np.histogram2d(X,Y,bins=[img.shape[0],img.shape[1]],range=[[0,img.shape[0]],[0,img.shape[1]]])[0]
        loc = np.where(img>threshold_otsu(img))
        self.utilities.save_data(loc,Dataset=self.dataset,Type='hot_pixels')

            
    def check_projection(self):
        if self.verbose:
            self.update_user('Checking Projection Zindexes')
        self.acq = [i for i in os.listdir(self.metadata_path) if 'hybe1_' in i][0]
        self.image_table = pd.read_csv(os.path.join(self.metadata_path,self.acq,'Metadata.txt'),sep='\t')
        self.len_z = len(self.image_table[(self.image_table.Position==self.posnames[0])].Zindex.unique())
        if self.projection_function=='None':
            self.projection_k = 0
        if self.projection_zstart==-1:
            self.projection_zstart = 0+self.projection_k
        elif self.projection_zstart>self.len_z:
            print('zstart of ',self.projection_zstart,' is larger than stk range of', self.len_z)
            raise(ValueError('Projection Error'))
        if self.projection_zend==-1:
            self.projection_zend = self.len_z-self.projection_k
        elif self.projection_zend>self.len_z:
            print('zend of ',self.projection_zend,' is larger than stk range of', self.len_z)
            raise(ValueError('Projection Error'))
        elif self.projection_zend<self.projection_zstart:
            print('zstart of ',self.projection_zstart,' is larger than zend of', self.projection_zend)
            raise(ValueError('Projection Error'))
        self.zindexes = np.array(range(self.projection_zstart,self.projection_zend,self.projection_zskip))
        if self.two_dimensional:
            self.zindexes = [0]

    def load_transcripts(self):
        if self.verbose:
            self.update_user('Loading Transcripts')
        """ IMPLEMENT"""
        self.check_projection()
        """ For all positions"""
        transcripts_full = []
        cell_metadata_full = []
        for posname in self.posnames:
            cell_metadata = self.fishdata.load_data('cell_metadata',dataset=self.dataset,posname=posname)
            if isinstance(cell_metadata,type(None)):
                continue
            cell_metadata['posname'] = posname
            cell_metadata_full.append(cell_metadata)
            """ for all zindexes"""
            for zindex in self.zindexes:
                """ Load Transcripts"""
                transcripts = self.fishdata.load_data('spotcalls',dataset=self.dataset,posname=posname,zindex=zindex)
                if isinstance(transcripts,type(None)):
                    continue
                elif len(transcripts)==0:
                    continue
                transcripts['zindex'] = zindex
                transcripts['posname'] = posname
                transcripts['cell_id'] = [self.dataset+'_'+posname+'_cell_'+str(int(i)) for i in transcripts['cell_label']]
                transcripts_full.append(transcripts)
        if len(transcripts_full)==0:
            print('Error')
        self.transcripts = pd.concat(transcripts_full,ignore_index=True)
        self.cell_metadata = pd.concat(cell_metadata_full,ignore_index=True)

    def train_logistic(self):
        """ Train Logistic Regressor to find False Positives"""
        if self.verbose:
            self.update_user('Training Logistic')
        converter = {i:'True' for i in self.transcripts['cword_idx'].unique()}
        blank_indices = np.array([i for i,gene in enumerate(self.merfish_config.aids) if 'lank' in gene])
        for i in blank_indices:
            converter[i] = 'False'
        self.transcripts['X'] = [converter[i] for i in self.transcripts['cword_idx']]

        from sklearn.model_selection import train_test_split
        columns = self.parameters['logistic_columns']
        data = self.transcripts[columns]
        data_true = data.loc[data[data['X']=='True'].index]
        data_false = data.loc[data[data['X']=='False'].index]
        """ downsample to same size """
        s = np.min([data_true.shape[0],data_false.shape[0]])
        data_true_down = data_true.loc[np.random.choice(data_true.index,s,replace=False)]
        data_false_down = data_false.loc[np.random.choice(data_false.index,s,replace=False)]
        data_down = pd.concat([data_true_down,data_false_down])
        X_train, X_test, y_train, y_test = train_test_split(data_down.drop('X',axis=1),data_down['X'], test_size=0.30,random_state=101)
        from sklearn import preprocessing
        self.scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        from sklearn.linear_model import LogisticRegression
        self.logmodel = LogisticRegression(max_iter=1000)
        self.logmodel.fit(X_train_scaled,y_train)
        predictions = self.logmodel.predict(X_test_scaled)

        from sklearn.metrics import classification_report
        print(classification_report(y_test,predictions))
        self.save_models() 
        
    def apply_logistic(self):
        """ Apply Logistic Regressor to remove False Positives"""
        if self.verbose:
            self.update_user('Applying Logistic')
        columns = self.parameters['logistic_columns']
        data = self.transcripts[columns]
        self.load_models() 
        X = self.scaler.transform(data.drop('X',axis=1))
        predictions = self.logmodel.predict(X)
        probabilities = self.logmodel.predict_proba(X)[:,1] # make sure it is always this orientation
        self.transcripts['probabilities'] = probabilities
        self.transcripts['predicted_X'] = predictions
        """ Instead filter by probability to 5% FPR """
        column = 'probabilities'
        c = self.transcripts[column]
        vmin,mid,vmax = np.percentile(c,[1,50,99])
        threshes = np.linspace(vmin,vmax,100)
        X = self.transcripts.X
        for thresh in threshes:
            mask = c>thresh
            if np.sum(mask)==0:
                fpr = -1
            else:
                fpr = 10*Counter(X[mask])['False']/X[mask].shape[0]
            if fpr<0.05:
                break
        self.transcripts['logistic_thresh'] = thresh
        self.transcripts = self.transcripts[mask]
        # self.transcripts = self.transcripts[self.transcripts['predicted_X']=='True']
        self.transcripts['gene'] = np.array(self.merfish_config.aids)[self.transcripts.cword_idx]
        
    def generate_counts(self):
        """ Generate Counts Table """
        """ Update to anndata object"""
        if self.verbose:
            self.update_user('Generating Counts')
        import torch
        cells = self.transcripts.cell_id.unique()
        cell_index = {cell:i for i,cell in enumerate(cells)}
        genes = self.merfish_config.aids
        genes_index = {gene:i for i,gene in enumerate(genes)}

        counts = torch.zeros([len(cells),len(genes)],dtype=int)
        if self.verbose:
            iterable = tqdm(enumerate(genes),total=len(genes),desc='Generating Counts')
        else:
            iterable = enumerate(genes)
        for i,gene in iterable:
            mask = self.transcripts.gene==gene
            for cell,cc in Counter(self.transcripts[mask].cell_id).items():
                counts[cell_index[cell],genes_index[gene]] = cc
        self.counts = pd.DataFrame(counts.numpy(),columns=self.merfish_config.aids,index=cells)
        self.cell_metadata.index = np.array(self.cell_metadata['cell_id'])
        common_cells = set(list(self.counts.index)).intersection(list(self.cell_metadata.index))
        self.counts = self.counts.loc[common_cells]
        self.cell_metadata = self.cell_metadata.loc[common_cells]
        """ Add Stage Coordinates"""
        pos_locations = {pos:self.metadata.image_table[self.metadata.image_table.Position==pos].XY.iloc[0] for pos in self.posnames}
        self.cell_metadata['posname_stage_x'] = [pos_locations[pos][0] for pos in self.cell_metadata.posname]
        self.cell_metadata['posname_stage_y'] = [pos_locations[pos][1] for pos in self.cell_metadata.posname]
        pixel_size = self.parameters['pixel_size']
        if self.dataset in self.parameters['camera_direction_dict'].keys():
            self.parameters['camera_direction'] = self.parameters['camera_direction_dict'][self.dataset]
        else:
            self.parameters['camera_direction'] = self.parameters['camera_direction_dict']['default']
        if self.dataset in self.parameters['xy_flip_dict'].keys():
            self.parameters['xy_flip'] = self.parameters['xy_flip_dict'][self.dataset]
        else:
            self.parameters['xy_flip'] = self.parameters['xy_flip_dict']['default']
        if self.parameters['xy_flip']:
            stage_x = np.array(self.cell_metadata['posname_stage_x']) + self.parameters['camera_direction'][0]*pixel_size*np.array(self.cell_metadata['x_pixel'])
            stage_y = np.array(self.cell_metadata['posname_stage_y']) + self.parameters['camera_direction'][1]*pixel_size*np.array(self.cell_metadata['y_pixel'])
        else:
            stage_x = np.array(self.cell_metadata['posname_stage_x']) + self.parameters['camera_direction'][0]*pixel_size*np.array(self.cell_metadata['y_pixel'])
            stage_y = np.array(self.cell_metadata['posname_stage_y']) + self.parameters['camera_direction'][1]*pixel_size*np.array(self.cell_metadata['x_pixel'])
        self.cell_metadata['stage_x'] = stage_x
        self.cell_metadata['stage_y'] = stage_y
            
    def save_data(self):
        """ Save Data """
        if self.verbose:
            self.update_user('Saving Data')
        self.fishdata.add_and_save_data(self.transcripts,'spotcalls',dataset=self.dataset)
        self.fishdata.add_and_save_data(self.counts,'counts',dataset=self.dataset)
        self.fishdata.add_and_save_data(self.cell_metadata,'cell_metadata',dataset=self.dataset)
        self.completed=True
        self.utilities.save_data('Passed',Dataset=self.dataset,Type='flag')
            
    def save_models(self):
        """ Save Models """
        if self.verbose:
            self.update_user('Saving Models')
        self.utilities = Utilities_Class(self.parameters['utilities_path'])
        self.utilities.save_data(self.logmodel,Dataset=self.dataset,Type='models')

    def load_models(self):
        """ Load Models """
        if self.verbose:
            self.update_user('Loading Models')
        self.utilities = Utilities_Class(self.parameters['utilities_path'])
        self.logmodel = self.utilities.load_data(Dataset=self.dataset,Type='models')
        
        
    def process_transcripts(self):
        self.load_transcripts()
        self.train_logistic()
        self.apply_logistic()
        self.generate_counts()
        self.save_data()
        
    def calculate_zscore(self):
        for bit in range(len(self.bitmap)):
            readout,hybe,channel = self.bitmap[bit]
            zindex='all' #MOVE
            sample = 500 #MOVE

            # """ Check if it has already been created """
            # fname = self.utilities.load_data(Dataset=self.dataset,
            #                                  Hybe=hybe,
            #                                  Channel=channel,
            #                                  Zindex=zindex,
            #                                  Type='ZScore',
            #                                  filename_only=True)
            # if os.path.exists(fname):
            #     if self.verbose:
            #         self.update_user('bit'+str(bit)+' Zscore already exists')
            #         self.update_user(fname)
            #     """ Already Exists """
            #     continue
            image_fnames = np.array([i for i in os.listdir(self.fishdata.base_path) if 'image' in i])
            zindexes = np.unique([int(i.split('_')[-2]) for i in image_fnames])
            if zindex!='all':
                image_fnames = np.array([i for i in image_fnames if i.split('_')[-2]==str(zindex)])
            if channel!='all':
                image_fnames = np.array([i for i in image_fnames if i.split('_')[-3]==channel])
            if hybe!='all':
                image_fnames = np.array([i for i in image_fnames if i.split('_')[-4]==hybe])
            if image_fnames.shape[0]>sample:
                image_fnames = np.random.choice(image_fnames,sample,replace=False)
            else:
                if self.verbose:
                    self.update_user('bit'+str(bit))
                    self.update_user('Not Enough Images for ZScore Calculation')
                break
            out = []
            if self.verbose:
                iterable = tqdm(image_fnames,desc='Calculating ZScore Bit'+str(bit))
            else:
                iterable = image_fnames
            # UPDATE TO MULTIPROCESSING
            for i in iterable:
                f = os.path.join(self.fishdata.base_path,i)
                img = cv2.imread(f,-1)
                out.append(torch.tensor(img.astype(float)))
            out = torch.dstack(out)
            # If dim = None It will ravel and do 1D not 2D 
            # Add as a parameter? 
            out = torch.quantile(out,torch.tensor(np.array([0.25,0.5,0.75]).astype(float)),dim=2)
            """ Potential issues if STD is 0 """
            """ Maybe add a blur? """
            """ Save to Utilities """
            self.utilities.save_data(out,Dataset=self.dataset,Hybe=hybe,Channel=channel,Zindex=zindex,Type='ZScore')