from MERFISH_Objects.Hybe import *
from MERFISH_Objects.Registration import *
from MERFISH_Objects.Stack import *
from MERFISH_Objects.Image import *
from MERFISH_Objects.Deconvolution import *
from MERFISH_Objects.FISHData import *
from tqdm import tqdm
import dill as pickle
import time
import os
import numpy as np
from metadata import Metadata
from fish_results import HybeData
from datetime import datetime

class Position_Class(object):
    def __init__(self,
                 metadata_path,
                 dataset,
                 posname,
                 cword_config,
                 fresh=True,
                 wait=300,
                 verbose=False):
        self.metadata_path = metadata_path
        self.dataset = dataset
        self.posname = posname
        self.fresh = fresh
        self.verbose = verbose
        self.wait = wait
        
        self.cword_config = cword_config
        self.merfish_config = importlib.import_module(self.cword_config)
        self.parameters = self.merfish_config.parameters
        self.utilities_path = self.parameters['utilities_path']
        self.daemon_path = self.parameters['daemon_path']
        self.hybe_daemon_path = os.path.join(self.daemon_path,'hybe')
        if not os.path.exists(self.hybe_daemon_path):
            os.mkdir(self.hybe_daemon_path)
            os.mkdir(os.path.join(self.hybe_daemon_path,'input'))
            os.mkdir(os.path.join(self.hybe_daemon_path,'output'))
        self.bitmap = self.merfish_config.bitmap
        self.all_hybes = list(np.unique([hybe for seq,hybe,channel in self.bitmap]))
        self.hybe_fnames_list = [str(self.dataset+'_'+self.posname+'_'+hybe+'.pkl') for hybe in self.all_hybes]
        self.fishdata = FISHData(os.path.join(self.metadata_path,self.parameters['fishdata']))
        
        self.projection_zstart=self.parameters['projection_zstart'] 
        self.projection_k=self.parameters['projection_k']
        self.projection_zskip=self.parameters['projection_zskip'] 
        self.projection_zend=self.parameters['projection_zend']
        self.projection_function=self.parameters['projection_function']
        self.two_dimensional = self.parameters['two_dimensional']
        
        self.segmentation_completed = False
        self.hybes_completed = False
        self.classification_completed = False
        self.completed = False
        self.passed = True
        self.failed = False
        
    def run(self):
        self.check_flag()
        if not self.failed:
            self.main()
            
    def update_user(self,message):
        """ For User Display"""
        i = [i for i in tqdm([],desc=str(datetime.now().strftime("%H:%M:%S"))+' '+str(message))]
        
    def main(self):
        self.check_imaging()
        if len(self.nucstain)>0:
            self.check_segmentation()
        if len(self.imaged_hybes)>0:
            self.check_hybes()
        if self.hybes_completed:
            if self.segmentation_completed:
                self.check_classification()

    def check_flag(self):
        if self.verbose:
            self.update_user('Checking Flag')
        flag = self.fishdata.load_data('flag',dataset=self.dataset,posname=self.posname)
        if flag == 'Failed':
            self.failed = True
            self.completed = True
                    
    def check_imaging(self):
        if self.verbose:
            self.update_user('Checking Imaging')
        self.total_hybes = list(np.unique([hybe for seq,hybe,channel in self.bitmap]))
        self.started_acqs = [i for i in os.listdir(self.metadata_path) if ('hybe' in i)&(i.split('_')[0] in self.total_hybes)]
        self.nucstain = [i for i in os.listdir(self.metadata_path) if 'nucstain' in i]
        dictionary = {''.join([i+'_' for i in os.listdir(os.path.join(self.metadata_path,self.started_acqs[0],posname_directory))[0][4:].split('_')[:-5]])[:-1]:posname_directory for posname_directory in os.listdir(os.path.join(self.metadata_path,self.started_acqs[0])) if not 'Metadata' in posname_directory}
        
        self.acqs = [i for i in self.started_acqs if os.path.exists(os.path.join(self.metadata_path,i,dictionary[self.posname]))]
        self.imaged_hybes = [i.split('_')[0] for i in self.acqs if 'hybe' in i]
        self.not_imaged = list(np.unique([hybe for seq,hybe,channel in self.bitmap if not hybe in self.imaged_hybes]))

    def check_segmentation(self):
        """ check segmentation flag"""
        flag = self.fishdata.load_data('flag',dataset=self.dataset,
                                       posname=self.posname,
                                       channel=self.parameters['nucstain_channel'])
        if self.verbose:
            self.update_user('Checking Segmentation')
        if flag == 'Started':
            do = 'nothing'
        elif flag =='Failed':
            self.fishdata.add_and_save_data('Failed','flag',
                                            dataset=self.dataset,
                                            posname=self.posname)
            self.fishdata.add_and_save_data('Segmentation Failed','log',
                                            dataset=self.dataset,
                                            posname=self.posname)
        elif flag =='Passed':
            self.segmentation_completed = True
        else:
            self.create_segmentation()
        
    def create_segmentation(self):
        """create segmentation object"""
        self.segmentation_daemon_path = os.path.join(self.daemon_path,'segmentation')
        if not os.path.exists(self.segmentation_daemon_path):
                os.mkdir(self.segmentation_daemon_path)
                os.mkdir(os.path.join(self.segmentation_daemon_path,'input'))
                os.mkdir(os.path.join(self.segmentation_daemon_path,'output'))
        fname = self.dataset+'_'+self.posname+'.pkl'
        fname_path = os.path.join(self.segmentation_daemon_path,'input',fname)
        data = {'metadata_path':self.metadata_path,
                    'dataset':self.dataset,
                    'posname':self.posname,
                    'cword_config':self.cword_config,
                    'level':'segmentation'}
        pickle.dump(data,open(fname_path,'wb'))
        self.fishdata.add_and_save_data('Started','flag',
                                            dataset=self.dataset,
                                            posname=self.posname,
                                            channel=self.parameters['nucstain_channel'])
        
    def check_hybes(self):
        if self.verbose:
            iterable = tqdm(self.imaged_hybes,desc=str(datetime.now().strftime("%H:%M:%S"))+'Checking Hybe Flags')
        else:
            iterable = self.imaged_hybes
        self.started = []
        self.passed = []
        self.not_started = []
        self.failed = []
        for hybe in iterable:
            flag =  self.fishdata.load_data('flag',dataset=self.dataset,
                                            posname=self.posname,hybe=hybe)
            if isinstance(flag,type(None)):
                self.not_started.append(hybe)
            elif flag == 'Started':
                self.started.append(hybe)
            elif flag == 'Passed':
                self.passed.append(hybe)
            elif flag =='Failed':
                self.failed.append(hybe)
                self.fishdata.add_and_save_data('Failed','flag',
                                                        dataset=self.dataset,
                                                        posname=self.posname)
                self.fishdata.add_and_save_data(str(hybe+' Failed'),'log',
                                                        dataset=self.dataset,
                                                        posname=self.posname)
        if not self.completed:
            if len(self.not_started)==0: # All hybes have been started
                if len(self.started)==0: # All imaged hybes have been completed
                    if len(self.not_imaged)==0:# All hybes have been imaged
                        if len(self.failed)==0: # No hybes failed
                            self.hybes_completed = True
            else:
                self.create_hybes()
                        
    def create_hybes(self):
        if self.verbose:
            iterable = tqdm(self.not_started,desc=str(datetime.now().strftime("%H:%M:%S"))+'Creating Hybes')
        else:
            iterable = self.not_started
        self.hybe_daemon_path = os.path.join(self.daemon_path,'hybe')
        if not os.path.exists(self.hybe_daemon_path):
                os.mkdir(self.hybe_daemon_path)
                os.mkdir(os.path.join(self.hybe_daemon_path,'input'))
                os.mkdir(os.path.join(self.hybe_daemon_path,'output'))
        for hybe in iterable:
            fname = self.dataset+'_'+self.posname+'_'+hybe+'.pkl'
            fname_path = os.path.join(self.hybe_daemon_path,'input',fname)
            data = {'metadata_path':self.metadata_path,
                    'dataset':self.dataset,
                    'posname':self.posname,
                    'hybe':hybe,
                    'cword_config':self.cword_config,
                    'level':'hybe'}
            pickle.dump(data,open(fname_path,'wb'))
            self.fishdata.add_and_save_data('Started','flag',
                                                        dataset=self.dataset,
                                                        posname=self.posname,
                                                        hybe=hybe)
        
    def check_projection(self):
        if self.verbose:
            self.update_user('Checking Projection Zindexes')
#         self.metadata = Metadata(os.path.join(self.metadata_path,self.acq))
        self.image_table = pd.read_csv(os.path.join(self.metadata_path,self.acqs[0],'Metadata.txt'),sep='\t')
        self.len_z = len(self.image_table[(self.image_table.Position==self.posname)].Zindex.unique())
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
        
    def check_classification(self):
        """ Calculate Zindexes """
        self.check_projection()
        """ Check Zindexes """
        if self.verbose:
            iterable = tqdm(self.zindexes,desc=str(datetime.now().strftime("%H:%M:%S"))+'Checking Classification Flag')
        else:
            iterable = self.zindexes
        self.started = []
        self.passed = []
        self.not_started = []
        self.failed = []
        for zindex in iterable:
            flag =  self.fishdata.load_data('flag',
                                            dataset=self.dataset,
                                            posname=self.posname,
                                            zindex=zindex)
            if isinstance(flag,type(None)):
                self.not_started.append(zindex)
            elif flag == 'Started':
                self.started.append(zindex)
            elif flag == 'Passed':
                self.passed.append(zindex)
            elif flag =='Failed':
                """ FIX What to do if a zindex fails"""
                self.failed.append(zindex)
        if not self.completed:
            if len(self.not_started)==0: # All zindex have been started
                if len(self.started)==0: # All zindex have been completed
                    if len(self.not_imaged)==0:# All zindex have been processed
                        self.zindex_completed = True
                        """ All Zindexes failed """
                        self.completed = True
                        if len(self.failed)==len(self.zindexes):
                            """ Add Log"""
                            self.fishdata.add_and_save_data('No Transcripts Detected',
                                                            'log',
                                                            dataset=self.dataset,
                                                            posname=self.posname)
                            """ Update flag"""
                            self.fishdata.add_and_save_data('Failed',
                                                            'flag',
                                                            dataset=self.dataset,
                                                            posname=self.posname)
                        else:
                            self.fishdata.add_and_save_data('Passed','flag',
                                            dataset=self.dataset,
                                            posname=self.posname)
            else:
                self.create_classification()
                        
    def create_classification(self):
        if self.verbose:
            iterable = tqdm(self.not_started,desc=str(datetime.now().strftime("%H:%M:%S"))+'Creating Classify')
        else:
            iterable = self.not_started
        self.classify_daemon_path = os.path.join(self.daemon_path,'classification')
        if not os.path.exists(self.classify_daemon_path):
                os.mkdir(self.classify_daemon_path)
                os.mkdir(os.path.join(self.classify_daemon_path,'input'))
                os.mkdir(os.path.join(self.classify_daemon_path,'output'))
        for zindex in iterable:
            fname = self.dataset+'_'+self.posname+'_'+str(zindex)+'.pkl'
            fname_path = os.path.join(self.classify_daemon_path,'input',fname)
            data = {'metadata_path':self.metadata_path,
                    'dataset':self.dataset,
                    'posname':self.posname,
                    'zindex':zindex,
                    'cword_config':self.cword_config,
                    'level':'classification'}
            pickle.dump(data,open(fname_path,'wb'))
            self.fishdata.add_and_save_data('Started','flag',
                                                        dataset=self.dataset,
                                                        posname=self.posname,
                                                        zindex=zindex)