##### from tqdm import tqdm
from metadata import Metadata
from MERFISH_Objects.Registration import *
from MERFISH_Objects.Stack import *
from MERFISH_Objects.Image import *
from hybescope_config.microscope_config import *
from MERFISH_Objects.FISHData import *
import dill as pickle
import os
import importlib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import time
import torch
from skimage.measure import regionprops, label
from sklearn.preprocessing import normalize
import time
import trackpy as tp
from tqdm import trange
import multiprocessing
from datetime import datetime


class Classify_Class(object):
    def __init__(self,
                 metadata_path,
                 dataset,
                 posname,
                 zindex,
                 cword_config,
                 verbose=False):
        self.metadata_path = metadata_path
        self.dataset = dataset
        self.posname = posname
        self.zindex = zindex
        self.verbose = verbose
        self.cword_config = cword_config
        
        self.merfish_config = importlib.import_module(self.cword_config)
        self.parameters =  self.merfish_config.parameters
        self.bitmap = self.merfish_config.bitmap
        # self.utilities = Utilities_Class(self.parameters['utilities_path'])
        self.fishdata = FISHData(os.path.join(self.metadata_path,self.parameters['fishdata']))
        # FIX
        self.parameters['overwrite_spots'] = False
        self.parameters['classify_logistic'] = 'dataset'
        self.passed = True
        self.completed = False
        
        if self.dataset in self.parameters['spot_parameters'].keys():
            self.parameters['spot_max_distance'] = self.parameters['spot_parameters'][self.dataset]['spot_max_distance']
            self.parameters['spot_minmass'] = self.parameters['spot_parameters'][self.dataset]['spot_minmass']
            self.parameters['spot_diameter'] = self.parameters['spot_parameters'][self.dataset]['spot_diameter']
            self.parameters['spot_separation'] = self.parameters['spot_parameters'][self.dataset]['spot_separation']
        else:
            self.parameters['spot_max_distance'] = self.parameters['spot_parameters']['default']['spot_max_distance']
            self.parameters['spot_minmass'] = self.parameters['spot_parameters']['default']['spot_minmass']
            self.parameters['spot_diameter'] = self.parameters['spot_parameters']['default']['spot_diameter']
            self.parameters['spot_separation'] = self.parameters['spot_parameters']['default']['spot_separation']

    def run(self):
        if self.parameters['overwrite_spots']:
            self.main()
        else:
            self.check_flags()
        
    def check_flags(self):
        """ Check if Position is fine before attempting Zindex """
        """FIX"""
        self.passed = True
        self.main()
        
    def main(self):
        """ Main Functionality """
        if not self.parameters['classification_overwrite']:
            self.load_data()
        if not self.completed:
            if not self.parameters['image_call_spots']:
                self.call_spots()
            else:
                self.generate_spots()
            if self.passed:
                self.pair_spots()
            if self.passed:
                self.build_barcodes()
            if self.passed:
                self.assign_codewords()
            if self.passed:
                self.collapse_spots()
            if self.passed:
                self.assign_cells()
                # self.generate_counts()
                self.save_data()
                
    def update_user(self,message):
        """ For User Display"""
        i = [i for i in tqdm([],desc=str(datetime.now().strftime("%H:%M:%S"))+' '+str(message))]
        
    def load_data(self):
        """ Check If Zindex is completed"""
        if self.verbose:
            self.update_user('Loading Data')
        self.transcripts = self.fishdata.load_data('spotcalls',
                                                   dataset=self.dataset,
                                                   posname=self.posname,
                                                   zindex=self.zindex)
        self.counts = self.fishdata.load_data('counts',
                                              dataset=self.dataset,
                                              posname=self.posname,
                                              zindex=self.zindex)

        if isinstance(self.transcripts,type(None))==False:
            self.completed = True
            # self.utilities.save_data('Passed',
            #                         Dataset=self.dataset,
            #                         Position=self.posname,
            #                         Zindex=self.zindex,
            #                         Type='flag')
        
    def load_spots(self,hybe,channel):
        """ Load Spots Previously Detected """
        spots = self.fishdata.load_data('spotcalls',
                                        dataset=self.dataset,
                                        posname=self.posname,
                                        hybe=hybe,
                                        channel=channel,
                                        zindex=self.zindex)
        return spots
    
    def call_spots(self):
        """ For each Bit """
        self.stk = ''
        spots_out = []
        if self.verbose:
            iterable = tqdm(range(len(self.bitmap)),desc='Calling Spots')
        else:
            iterable = range(len(self.bitmap))
        for bit in iterable:
            readout,hybe,channel = self.bitmap[bit]
            """ Load Images """
            img = self.fishdata.load_data('image',
                                     dataset=self.dataset,
                                     posname=self.posname,
                                     hybe=hybe,
                                     channel=channel,
                                     zindex=self.zindex)
            if isinstance(img,type(None)):
                """ Image Doesnt Exist"""
                """ Should this error?"""
                # self.passed=False
                break         
            else:
                img = img.astype(float)
                img = img/self.parameters['gain'] # Restore Scale
                if isinstance(self.stk,str):
                    self.stk = np.zeros([img.shape[0],img.shape[1],len(self.bitmap)]).astype(float)
                """ ZScore Global"""
                # temp = self.utilities.load_data(Dataset=self.dataset,
                #                                 Hybe=hybe,
                #                                 Channel=channel,
                #                                 Zindex='all',
                #                                 Type='ZScore')
                temp = None
                if isinstance(temp,type(None)):
                    """ Wait For ZScore Properties"""
                    temp = np.percentile(img.ravel(),[25,50,75])
                    median = temp[1]
                    zscore = img-median
                    std = temp[2]-temp[0]
                    zscore = zscore/std
                else:
                    median = temp[1,:,:].numpy()
                    zscore = img-median
                    std = temp[2,:,:].numpy()-temp[0,:,:].numpy()
                    zscore = zscore/std
                """ Check for nans?"""
                zscore[std==0] = 0
                # zscore[zscore<0] = 0
                self.stk[:,:,bit] = zscore
                """ Call Spots """
                spots = tp.locate(zscore,
                                   self.parameters['spot_diameter'],
                                   minmass=self.parameters['spot_minmass'],
                                   separation=self.parameters['spot_separation'])
                self.fishdata.add_and_save_data(spots,
                                                'spotcalls',
                                                dataset=self.dataset,
                                                posname=self.posname,
                                                hybe=hybe,
                                                channel=channel,
                                                zindex=self.zindex)
                if not isinstance(spots,type(None)):
                    if len(spots)>0:
                        spots['bit'] = bit
                        spots['zindex'] = self.zindex
                        spots_out.append(spots)
        if self.passed:
            if len(spots_out)==0:
                """ Error No Spots Detected"""
                self.passed = False
                self.transcripts = None
                self.completed = True

#                 self.utilities.save_data('Failed',
#                                         Dataset=self.dataset,
#                                         Position=self.posname,
#                                         Zindex=self.zindex,
#                                         Type='flag')
#                 self.utilities.save_data('No Spots Detected',
#                                         Dataset=self.dataset,
#                                         Position=self.posname,
#                                         Zindex=self.zindex,
#                                         Type='log')
            else:
                self.spots = pd.concat(spots_out,ignore_index=True)
        
    def generate_spots(self):
        """ Generate Spots for all bits for this Zindex """
        if self.verbose:
            iterable = tqdm(enumerate(self.merfish_config.bitmap),total=len(self.merfish_config.bitmap),desc='Loading Spots')
        else:
            iterable = enumerate(self.merfish_config.bitmap)
        spots_out = []
        for i,(readout,hybe,channel) in iterable:
            if not self.passed:
                continue
            spots = self.load_spots(hybe,channel)
            if not isinstance(spots,type(None)):
                if len(spots)>0:
                    spots['bit'] = i
                    spots['zindex'] = self.zindex
                    spots_out.append(spots)
        if len(spots_out)==0:
            """ Error No Spots Detected"""
            self.passed = False
            self.transcripts = None
            self.completed = True
            
#             self.utilities.save_data('Failed',
#                                     Dataset=self.dataset,
#                                     Position=self.posname,
#                                     Zindex=self.zindex,
#                                     Type='flag')
#             self.utilities.save_data('No Spots Detected',
#                                     Dataset=self.dataset,
#                                     Position=self.posname,
#                                     Zindex=self.zindex,
#                                     Type='log')

        else:
            self.spots = pd.concat(spots_out,ignore_index=True)
        
    def pair_spots(self):
        """ Pair spots to transcripts"""
        if self.verbose:
            self.update_user('Pairing spots')
        X = np.zeros([self.spots.shape[0],3])
        X[:,0] = self.spots.x.astype(float)*self.parameters['pixel_size']
        X[:,1] = self.spots.y.astype(float)*self.parameters['pixel_size']
        X[:,2] = self.spots.zindex.astype(float)*self.merfish_config.parameters['z_step_size']
        clustering = DBSCAN(eps=self.parameters['spot_max_distance']*self.parameters['pixel_size'], min_samples=3).fit(X)
        self.spots['label'] = clustering.labels_
        good_labels = [i for i,c in Counter(clustering.labels_).items() if c<6]
        self.spots = self.spots[self.spots['label'].isin(good_labels)]
        if self.spots.shape[0]==0:
            """ Error No Spots Detected"""
            self.passed = False
            self.transcripts = None
            self.completed = True
            # self.utilities.save_data('Failed',
            #                         Dataset=self.dataset,
            #                         Position=self.posname,
            #                         Zindex=self.zindex,
            #                         Type='flag')
            # self.utilities.save_data('No Spots Paired',
            #                         Dataset=self.dataset,
            #                         Position=self.posname,
            #                         Zindex=self.zindex,
            #                         Type='log')
        
    def build_barcodes(self):
        """ Build Barcode from paired spots"""
        if self.verbose:
            self.update_user('Building Barcodes')
        self.measured_barcodes = torch.zeros([np.unique(self.spots['label']).shape[0],self.merfish_config.nbits])
        labels_converter = {j:i for i,j  in enumerate(np.unique(self.spots['label']))}
        self.spots['idx'] = [labels_converter[i] for i in self.spots['label']]
        self.measured_barcodes[np.array(self.spots['idx']).astype(int),np.array(self.spots['bit']).astype(int)] = 1
    
    def assign_codewords(self):
        """ Assign codeword to transcripts """
        if self.verbose:
            self.update_user('Assigning Codewords')
        """ Decode """
        self.barcodes = torch.tensor(self.merfish_config.all_codeword_vectors.astype(float))
        values,cwords = torch.cdist(self.measured_barcodes.float(),self.barcodes.float()).min(1)
        values = values**2 # Return to bitwise distance
        values = torch.tensor([round(i) for i in values.numpy()])
        """ Filter to good decoded """
        mask = values<2
        self.good_values = values[mask]
        self.good_cwords = cwords[mask]
        self.good_indices = torch.tensor(np.array(range(self.measured_barcodes.shape[0])))[mask]
        if self.good_indices.shape[0]==0:
            """ Error No Spots Detected"""
            self.passed = False
            self.transcripts = None
            self.completed = True
            # self.utilities.save_data('Failed',
            #                         Dataset=self.dataset,
            #                         Position=self.posname,
            #                         Zindex=self.zindex,
            #                         Type='flag')
            # self.utilities.save_data('No Codewords Assigned',
            #                         Dataset=self.dataset,
            #                         Position=self.posname,
            #                         Zindex=self.zindex,
            #                         Type='log')
        else:
            self.spots = self.spots[self.spots['idx'].isin(list(self.good_indices.numpy()))]
        
    def collapse_spots(self):
        """ Generate transcript dataframe"""
        if self.verbose:
            self.update_user('Collapsing Spots')
        """ Collapse Spots to Counts """
        transcripts = []
        for i,idx in enumerate(list(self.good_indices.numpy())):
            """ What information should be passed along to logistic regressor"""
            mask = self.spots['idx']==idx
            if np.sum(mask)==0:
                continue
            temp = self.spots[mask]
            positive_bits = np.array(list(temp['bit']))
            cword_idx = self.good_cwords[i].numpy()
            expected_positive_bits = torch.where(self.barcodes[cword_idx]==1)[0].numpy()
            negative_bits = np.array(range(len(self.bitmap)))
            negative_bits = negative_bits[np.isin(negative_bits,expected_positive_bits)==False]
            # remove False positives before averaging?
            dispersion = (self.parameters['pixel_size']*temp.x.std())+(self.parameters['pixel_size']*temp.y.std())+(self.parameters['z_step_size']*temp.zindex.std())
            temp = temp.mean()
            temp['intensity'] = temp['signal']
            temp['dispersion'] = dispersion
            """ Vector of Signal?"""
            V = self.stk[int(temp.y),int(temp.x),:]
            pos_signal = np.mean(V[expected_positive_bits])
            neg_signal = np.mean(V[negative_bits])
            correct_bits = int(0)
            false_negatives = int(0)
            false_positives = int(0)
            for b in range(len(self.bitmap)):
                if b in expected_positive_bits:
                    if b in positive_bits:
                        correct_bits+=1
                    else:
                        false_negatives+=1
                else:
                    if b in positive_bits:
                        false_positives+=1
                t = V[b].mean()
                temp['bit'+str(b)] = t
            temp['correct_bits'] = correct_bits
            temp['false_positives'] = false_positives
            temp['false_negatives'] = false_negatives
            temp['signal'] = pos_signal
            temp['noise'] = neg_signal
            temp['signal-noise'] = pos_signal-neg_signal
            temp['n_spots'] = np.sum(mask)
            temp['cword_idx'] = cword_idx
            temp['cword_distance'] = self.good_values[i].numpy()
            transcripts.append(pd.DataFrame(temp).T)
        transcripts = pd.concat(transcripts,ignore_index=True)
        transcripts = transcripts.dropna()
        transcripts = transcripts.drop(columns=['bit'])
        transcripts['cword_distance'] = [int(round(float(i))) for i in transcripts['cword_distance']]
        transcripts['cword_idx'] = [int(round(float(i))) for i in transcripts['cword_idx']]
        self.transcripts = transcripts 
        
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
        columns = self.parameters['logistic_columns']
        data = self.transcripts[columns]
        self.load_models() 
        predictions = self.logmodel.predict(data.drop('X',axis=1))
        self.transcripts['predicted_X'] = predictions
        self.transcripts = self.transcripts[self.transcripts['predicted_X']=='True']
        self.transcripts['gene'] = np.array(self.merfish_config.aids)[self.transcripts.cword_idx]
        
    def load_segmentation(self):
        """ Load Segmentation for Transcript assignment"""
        if self.verbose:
            self.update_user('Loading Segmentation')
        if self.parameters['segment_two_dimensional']:
            self.cytoplasm_mask = self.fishdata.load_data('cytoplasm_mask',dataset=self.dataset,posname=self.posname)
            self.nuclei_mask = self.fishdata.load_data('nuclei_mask',dataset=self.dataset,posname=self.posname)
            self.cell_metadata = self.fishdata.load_data('cell_metadata',dataset=self.dataset,posname=self.posname)
        else:
            self.cytoplasm_mask = self.fishdata.load_data('cytoplasm_mask',dataset=self.dataset,posname=self.posname,zindex=self.zindex)
            self.nuclei_mask = self.fishdata.load_data('nuclei_mask',dataset=self.dataset,posname=self.posname,zindex=self.zindex)
            self.cell_metadata = self.fishdata.load_data('cell_metadata',dataset=self.dataset,posname=self.posname,zindex=self.zindex)
        if isinstance(self.nuclei_mask,type(None)):
            """ Issue """
            self.passed = False
        if isinstance(self.nuclei_mask,type(None)):
            """ Issue """
            self.passed = False
        if self.cytoplasm_mask.max()==0:
            if self.verbose:
                print('No Cells in '+self.posname+' zindex: '+str(self.zindex))
                """ Fail Position"""
                
    def assign_cells(self):
        """ Assign transcripts to cells"""
        self.load_segmentation()
        pixel_x = np.array(self.transcripts.x).astype(int)
        pixel_y = np.array(self.transcripts.y).astype(int)
        cell_labels = self.cytoplasm_mask[pixel_y,pixel_x]
        nuclei_labels = self.nuclei_mask[pixel_y,pixel_x]
        self.transcripts['posname'] = self.posname
        self.transcripts['cell_label'] = cell_labels
        self.transcripts['nuclei_label'] = nuclei_labels
        
    # def generate_counts(self):
    #     """ Generate Counts Table """
    #     """ Update to anndata object"""
    #     if self.verbose:
    #         self.update_user('Generating Counts')
    #     self.assignCells()
    #     cells = self.transcripts.cell_label.unique()
    #     counts = np.zeros([len(cells),len(self.merfish_config.aids)],dtype=int)
    #     if self.verbose:
    #         iterable = tqdm(enumerate(cells),total=len(cells),desc='Generating Counts')
    #     else:
    #         iterable = enumerate(cells)
    #     for i,cell in iterable:
    #         for cword_idx,cc in Counter(self.transcripts[self.transcripts.cell_label==cell].cword_idx).items():
    #             counts[i,cword_idx] = cc
    #     self.counts = pd.DataFrame(counts,columns=self.merfish_config.aids,index=cells)
            
    def save_data(self):
        """ Save Data """
        if self.verbose:
            self.update_user('Saving Data')
        self.fishdata.add_and_save_data(self.transcripts,'spotcalls',dataset=self.dataset,posname=self.posname,zindex=self.zindex)
        # self.utilities.save_data('Passed',
        #                             Dataset=self.dataset,
        #                             Position=self.posname,
        #                             Zindex=self.zindex,
        #                             Type='flag')
        # self.fishdata.add_and_save_data(self.counts,'counts',dataset=self.dataset,posname=self.posname,zindex=self.zindex)
            
    def save_models(self):
        """ Save Models """
        if self.verbose:
            self.update_user('Saving Models')
        # self.utilities = Utilities_Class(self.utilities_path)
        self.utilities.save_data(self.logmodel,Dataset=self.dataset,Type='models')

    def load_models(self):
        """ Load Models """
        if self.verbose:
            self.update_user('Loaing Models')
        # self.utilities = Utilities_Class(self.utilities_path)
        # self.logmodel = self.utilities.load_data(Dataset=self.dataset,Type='models')
        
