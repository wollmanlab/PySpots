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
                 wait = 300,
                 verbose=True):
        self.metadata_path = metadata_path
        self.dataset = dataset
        self.cword_config = cword_config
        self.verbose = verbose
        self.wait = wait
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
        self.flag = self.fishdata.add_and_save_data('Started','flag',dataset=self.dataset)
        if self.verbose:
            iterable = tqdm(self.posnames,desc=str(datetime.now().strftime("%H:%M:%S"))+' Checking Position Flags')
        else:
            iterable = self.posnames
        self.started = []
        self.passed = []
        self.not_started = []
        self.failed = []
        for posname in iterable:
            flag =  self.fishdata.load_data('flag',dataset=self.dataset,posname=posname)
            if isinstance(flag,type(None)):
                self.not_started.append(posname)
            elif flag == 'Started':
                self.started.append(posname)
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
            self.fishdata.add_and_save_data('Started','flag',
                                                        dataset=self.dataset,
                                                        posname=posname)
            
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
        columns = ['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep', 'cword_distance','X']
        data = self.transcripts[columns]
        data_true = data.loc[data[data['X']=='True'].index]
        data_false = data.loc[data[data['X']=='False'].index]
        """ downsample to same size """
        s = np.min([data_true.shape[0],data_false.shape[0]])
        data_true_down = data_true.loc[np.random.choice(data_true.index,s,replace=False)]
        data_false_down = data_false.loc[np.random.choice(data_false.index,s,replace=False)]
        data_down = pd.concat([data_true_down,data_false_down])
        X_train, X_test, y_train, y_test = train_test_split(data_down.drop('X',axis=1),data_down['X'], test_size=0.30,random_state=101)

        from sklearn.linear_model import LogisticRegression
        self.logmodel = LogisticRegression(max_iter=1000)
        self.logmodel.fit(X_train,y_train)
        predictions = self.logmodel.predict(X_test)

        from sklearn.metrics import classification_report
        print(classification_report(y_test,predictions))
        self.save_models() 
        
    def apply_logistic(self):
        """ Apply Logistic Regressor to remove False Positives"""
        if self.verbose:
            self.update_user('Applying Logistic')
        columns = ['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep', 'cword_distance','X']
        data = self.transcripts[columns]
        self.load_models() 
        predictions = self.logmodel.predict(data.drop('X',axis=1))
        self.transcripts['predicted_X'] = predictions
        self.transcripts = self.transcripts[self.transcripts['predicted_X']=='True']
        self.transcripts['gene'] = np.array(self.merfish_config.aids)[self.transcripts.cword_idx]
        
    def generate_counts(self):
        """ Generate Counts Table """
        """ Update to anndata object"""
        if self.verbose:
            self.update_user('Generating Counts')
        cells = self.transcripts.cell_id.unique()
        counts = np.zeros([len(cells),len(self.merfish_config.aids)],dtype=int)
        if self.verbose:
            iterable = tqdm(enumerate(cells),total=len(cells),desc='Generating Counts')
        else:
            iterable = enumerate(cells)
        for i,cell in iterable:
            for cword_idx,cc in Counter(self.transcripts[self.transcripts.cell_id==cell].cword_idx).items():
                counts[i,cword_idx] = cc
        self.counts = pd.DataFrame(counts,columns=self.merfish_config.aids,index=cells)
            
    def save_data(self):
        """ Save Data """
        if self.verbose:
            self.update_user('Saving Data')
        self.fishdata.add_and_save_data(self.transcripts,'spotcalls',dataset=self.dataset)
        self.fishdata.add_and_save_data(self.counts,'counts',dataset=self.dataset)
        self.fishdata.add_and_save_data(self.cell_metadata,'cell_metadata',dataset=self.dataset)
        self.completed=True
        self.fishdata.add_and_save_data('Passed','flag',dataset=self.dataset)
            
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