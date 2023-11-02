##### from tqdm import tqdm
from metadata import Metadata
from MERFISH_Objects.Registration import *
from MERFISH_Objects.Stack import *
from MERFISH_Objects.Image import *
from MERFISH_Objects.hybescope_config.microscope_config import *
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
        """ Class to Detect/Process smFISH Spots to MERFISH Transcripts

        Args:
            metadata_path (str): Path to Raw Dataset
            dataset (str): Name of Dataset
            posname (str): Name of Position
            zindex (str): Name of Zindex
            cword_config (str): Name of Config Module
            verbose (bool, optional): _description_. Defaults to False.
        """
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
        """ Main Executable For Asyncronous Data Pipeline 
        """
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
        classification_method = self.merfish_config.parameters['classification_method'] 
        if classification_method == 'spot':
            self.spot_based()
        elif classification_method == 'pixel':
            self.pixel_based()
        else:
            self.update_user('Invalid classification_method: '+classification_method)
                
    def update_user(self,message):
        """ For User Display

        Args:
            message (str): will be sent to user
        """
        if self.verbose:
            i = [i for i in tqdm([],desc=str(datetime.now().strftime("%H:%M:%S"))+' '+str(message))]

    def spot_based(self):
        if not self.parameters['classification_overwrite']:
            self.load_data()
        if not self.completed:
            if not self.parameters['image_call_spots']:
                self.call_spots()
            else:
                self.generate_spots()
            for iteration in range(self.parameters['classify_iterations']):
                if self.passed:
                    self.pair_spots()
                if self.passed:
                    self.build_barcodes()
                if self.passed:
                    self.assign_codewords()
                if self.passed:
                    self.collapse_spots()
                if self.passed:
                    self.transcripts = self.predict_real_spots(self.transcripts,self.transcripts,columns=self.parameters['logistic_columns'],label='cword_idx')
                    #self.logistic_filter()
            if self.passed:
                self.assign_cells()
                self.save_data()        

    def pixel_based(self):
        # Load Data
        self.load_data()
        # Load Stack
        self.load_stk()
        if isinstance(self.stk,str):
            self.update_user('No images')
        elif np.sum(self.stk)==0:
            self.update_user('Empty Images')
        else:
            # Convert to Tensor
            self.update_user('Converting To Tensor')
            stk = torch.tensor(self.stk)
            barcodes = torch.tensor(self.merfish_config.all_codeword_vectors)
            # Define ROI
            self.update_user('Defining Region of Interese')
            thresh = 4 #ZSCORE
            roi_mask = stk.max(2).values>thresh
            pixel_y,pixel_x = torch.where(roi_mask)
            # Pull vectors
            self.update_user('Pulling Vectors')
            vectors =  stk[roi_mask,:]
            # Normalize 
            self.update_user('Normalizing Vectors and Barcodes')
            norm_vectors = torch.tensor(normalize(vectors))
            norm_barcodes = torch.tensor(normalize(barcodes))
            # Assign Cwords
            self.update_user('Assigning Barcodes to Pixels')
            dist,idx = torch.cdist(norm_vectors,norm_barcodes).min(1)
            # Pull Signal and Noise 
            self.update_user('Pulling Pixel Metrics')
            vector_barcodes = barcodes[idx,:]
            signal = torch.clone(vectors)
            signal[vector_barcodes==0] = 0
            signal = signal.sum(1)/vector_barcodes.sum(1)
            noise = torch.clone(vectors)
            noise[vector_barcodes==1] = 0
            noise = noise.sum(1)/(vector_barcodes.shape[1]-vector_barcodes.sum(1))
            # Pull Signal and Noise 
            norm_signal = torch.clone(norm_vectors)
            norm_signal[vector_barcodes==0] = 0
            norm_signal = norm_signal.sum(1)/vector_barcodes.sum(1)
            norm_noise = torch.clone(norm_vectors)
            norm_noise[vector_barcodes==1] = 0
            norm_noise = norm_noise.sum(1)/(vector_barcodes.shape[1]-vector_barcodes.sum(1))
            # Assemble Pixels
            pixels = pd.DataFrame(idx.numpy(),columns=['cword_idx'])
            pixels['dist'] = dist.numpy()
            pixels['x'] = pixel_x.numpy()
            pixels['y'] = pixel_y.numpy()
            pixels['signal'] = signal.numpy()
            pixels['noise'] = noise.numpy()
            pixels['signal-noise'] = signal.numpy()-noise.numpy()
            pixels['norm_signal'] = norm_signal.numpy()
            pixels['norm_noise'] = norm_noise.numpy()
            pixels['norm_signal-norm_noise'] = norm_signal.numpy()-norm_noise.numpy()
            converter = {idx:str('lank' not in gene) for idx,gene in enumerate(self.merfish_config.aids)}
            pixels['X'] = [converter[idx] for idx in pixels['cword_idx']]
            pixels['gene'] = np.array(self.merfish_config.aids)[pixels['cword_idx']]
            filtered_pixels = pixels[(pixels['signal']>0)&(pixels['signal-noise']>0)&(pixels['dist']<0.7)].copy()
            if filtered_pixels.shape[0]>0:
                # Pair Pixels
                self.update_user('Merging Pixels')
                idx_map = -1*torch.tensor(np.ones([stk.shape[0],stk.shape[1]]).astype(int))
                idx_map[torch.tensor(np.array(filtered_pixels['y']).astype(int)),torch.tensor(np.array(filtered_pixels['x']).astype(int))] = torch.tensor(np.array(filtered_pixels['cword_idx']))
                from skimage.measure import regionprops, label
                label2d = torch.tensor(label((idx_map.numpy()+1).astype('uint16'), connectivity=2))
                filtered_pixels['transcript'] = label2d[torch.tensor(np.array(filtered_pixels['y']).astype(int)),torch.tensor(np.array(filtered_pixels['x']).astype(int))].numpy()
                converter = {idx:cc for idx,cc in Counter(filtered_pixels['transcript']).items()}
                filtered_pixels['n_pixels'] = [converter[idx] for idx in filtered_pixels['transcript']]
                # Create Transcripts
                self.update_user('Creating Transcripts')
                transcripts = filtered_pixels.groupby('transcript').mean()
                converter = {idx:str('lank' not in gene) for idx,gene in enumerate(self.merfish_config.aids)}
                transcripts['X'] = [converter[idx] for idx in transcripts['cword_idx'].astype(int)]
                transcripts['gene'] = np.array(self.merfish_config.aids)[transcripts['cword_idx'].astype(int)]
                # Filter
                pixels = transcripts.copy()
                # Define threshold
                self.update_user('Calculating Distance Threshold')
                correction = len(self.merfish_config.aids)/len(self.merfish_config.bids)
                c = pixels['dist']
                vmin,vmax = np.percentile(c,[1,99])
                bins = np.linspace(vmin,vmax,100)
                fpr = np.zeros_like(bins)
                for i,b in enumerate(bins):
                    filtered_pixels = pixels[pixels['dist']<b]
                    fpr[i] = correction*np.sum(filtered_pixels['X']=='False')/filtered_pixels.shape[0]
                    if fpr[i]>0.2: #HARDCODED
                        thresh = bins[i-1]
                        self.update_user('thresh: '+str(thresh))
                        self.update_user('FPR: '+str(fpr[i-1]))
                        break
                # Filter out false pixels
                self.update_user('Predicting False Pixels')
                columns = ['dist','signal','noise','signal-noise','norm_signal','norm_noise','norm_signal-norm_noise','n_pixels','X']
                training_data = pixels[pixels['dist']<thresh].copy()
                training_data = training_data[training_data['gene']!='Malat1'] # Better Balance Genes here ### HARDCODED
                application_data = pixels.copy()
                from sklearn.model_selection import train_test_split
                data = training_data[columns]
                data_true = data.loc[data[data['X']=='True'].index]
                data_false = data.loc[data[data['X']=='False'].index]
                if np.min([data_true.shape[0],data_false.shape[0]])>0:
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
                    from sklearn.neural_network import MLPClassifier
                    self.model  = MLPClassifier(alpha=1, max_iter=1000)
                    self.model.fit(X_train_scaled,y_train)
                    predictions = self.model.predict(X_test_scaled)
                    from sklearn.metrics import classification_report
                    if self.verbose:
                        print(classification_report(y_test,predictions))
                    data = application_data[[i for i in columns if not 'X'==i]]
                    X = self.scaler.transform(data)
                    pixels['predicted_X'] = self.model.predict(X)
                    pixels['probabilities_X'] = self.model.predict_proba(X)[:,1] 
                    filtered_pixels = pixels.copy()
                else:
                    self.update_user('Not enough Pixels to Train')
                    filtered_pixels = application_data

                self.transcripts = filtered_pixels
                # Assign Cells
                self.assign_cells()
                # Save Data
                self.save_data()
        
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

    def load_stk(self):
        self.stk = ''
        spots_out = []
        if self.verbose:
            iterable = tqdm(range(len(self.bitmap)),desc='Loading Stack')
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
                self.update_user('No Image Found Bit:'+str(bit))
                # break         
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
                    if std!=0:
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
        
    def load_spots(self,hybe,channel):
        """Load Spots Previously Detected

        Args:
            hybe (str): Name of Hybe
            channel (str): Name of Channel
        Returns:
            pd.DataFrame: dataframe of detected smFISH Spots
        """
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
                self.update_user('No Image Found Bit:'+str(bit))
                # img = self.stk[:,:,0].copy()*0 ### Zeros as filin
                # break         
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
                    if std!=0:
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
                                   self.parameters['spot_diameter'],separation=self.parameters['spot_separation'],
                                   preprocess=False)
                spots['intensity'] = spots['signal']
                spots = spots[spots['intensity']>np.percentile(zscore.ravel(),self.parameters['spot_percentile'])]
                spots = spots[spots['intensity']>self.parameters['spot_minmass']]
                # spots = tp.locate(zscore,
                #                self.parameters['spot_diameter'],
                #                percentile=self.parameters['spot_minmass'],
                #                separation=self.parameters['spot_separation']) 
                spots['spot_diameter'] = self.parameters['spot_diameter']
                spots['spot_minmass'] = self.parameters['spot_minmass']
                spots['spot_separation'] = self.parameters['spot_separation']
                spots['spot_max_distance'] = self.parameters['spot_max_distance']
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
                    else:
                        self.update_user('No Spots Found Bit:'+str(bit))
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
        if self.verbose:
            self.update_user(str(len(self.spots))+' Spots Found')
        
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
        if self.verbose:
            self.update_user(str(len(self.spots))+' Spots Found')
        
    def pair_spots(self):
        """ Pair spots to transcripts"""
        if self.verbose:
            self.update_user('Pairing spots')
        X = np.zeros([self.spots.shape[0],2])
        X[:,0] = self.spots.x.astype(float)*self.parameters['pixel_size']
        X[:,1] = self.spots.y.astype(float)*self.parameters['pixel_size']
        xy_labels,cluster_df = MNN_Agglomerative(X,max_distance = self.parameters['spot_max_distance'],verbose=self.verbose)
        if self.verbose:
            self.update_user('Pairing Results')
            for n_spots,n_transcripts in Counter([cc for idx,cc in Counter(xy_labels).items()]).items():
                print(str(n_transcripts)+' Pairs have '+str(n_spots)+' Spots')
            # print(Counter([cc for idx,cc in Counter(xy_labels).items()]).items())
        self.spots['label'] = xy_labels#clustering.labels_
        good_labels = [i for i,c in Counter(xy_labels).items() if (c<7)&(i!=0)&(c>1)]
        self.spots = self.spots[self.spots['label'].isin(good_labels)]
        if self.verbose:
            self.update_user(str(len(self.spots))+' Spots Remaining')
        if self.verbose:
            self.update_user(str(np.unique(self.spots['label']).shape[0])+' Potential Transcripts Found')
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

    def build_vectors(self):
        self.vectors = np.zeros_like(self.measured_barcodes)
        if self.verbose:
            iterable = tqdm(enumerate(list(np.unique(self.spots['idx']))),total=len(list(np.unique(self.spots['idx']))),desc='Building Vector')
        else:
            iterable = enumerate(list(np.unique(self.spots['idx'])))
        for i,idx in iterable:
            mask = self.spots['idx']==idx
            if np.sum(mask)==0:
                continue
            temp = self.spots[mask]
            V = self.stk[int(temp.y.mean()),int(temp.x.mean()),:] ### Pull positive bits from called spots 
            for spot,row in temp.iterrows():
                V[int(row['bit'])] = self.stk[int(row.y),int(row.x),int(row['bit'])]
            self.vectors[i,:] = V

    def update_binary(self):
        if self.verbose:
            self.update_user('Updating Binary')
        from sklearn.model_selection import train_test_split
        """ Update Binary with logistic """
        self.updated_measured_barcodes = np.zeros_like(self.measured_barcodes)
        for i in range(self.vectors.shape[1]):
            if self.vectors[:,i].max()==0:
                self.update_user('Skipping Empty: Bit '+str(i))
                self.barcodes[:,i] = 0
                continue
            if np.sum(self.measured_barcodes[:,i].numpy()==1)==0:
                self.update_user('Skipping No Spots: Bit '+str(i))
                # self.barcodes[:,i] = 0
                continue
            data = pd.DataFrame(self.vectors[:,i])
            data['X'] = self.measured_barcodes[:,i].numpy()==1
            mask = (self.measured_barcodes.sum(1).numpy()>2)&(self.measured_barcodes.sum(1).numpy()<6)
            data = data[mask]
            data_true = data.loc[data[data['X']==True].index]
            data_false = data.loc[data[data['X']==False].index]
            """ downsample to same size """
            s = np.min([data_true.shape[0],data_false.shape[0]])
            data_true_down = data_true.loc[np.random.choice(data_true.index,s,replace=False)]
            data_false_down = data_false.loc[np.random.choice(data_false.index,s,replace=False)]
            data_down = pd.concat([data_true_down,data_false_down])
            X_train, X_test, y_train, y_test = train_test_split(data_down.drop('X',axis=1),data_down['X'], test_size=0.30,random_state=101)
            from sklearn import preprocessing
            self.scaler = preprocessing.StandardScaler().fit(X_train)
            X_train_scaled = X_train#self.scaler.transform(X_train)
            X_test_scaled = X_test#self.scaler.transform(X_test)
            from sklearn.linear_model import LogisticRegression
            self.logmodel = LogisticRegression(max_iter=1000)
            self.logmodel.fit(X_train_scaled,y_train)
            # predictions = self.logmodel.predict(X_test_scaled)
            # from sklearn.metrics import classification_report
            # print(classification_report(y_test,predictions))
            predictions = self.logmodel.predict(pd.DataFrame(self.vectors[:,i]))
            self.updated_measured_barcodes[:,i] = 1*predictions
        self.updated_measured_barcodes = torch.tensor(self.updated_measured_barcodes)
        if self.verbose:
            self.update_user('Updating Binary Results:')
            dif = self.updated_measured_barcodes-self.measured_barcodes
            dif[dif>0] = 0
            print('Spots Lost per bit')
            print(dif.sum(0))
            dif = self.updated_measured_barcodes-self.measured_barcodes
            dif[dif<0] = 0
            print('Spots Gaines per bit')
            print(dif.sum(0))

    def assign_codewords(self):
        """ Assign codeword to transcripts """
        if self.verbose:
            self.update_user('Assigning Codewords')
        """ Decode """
        self.barcodes = torch.tensor(self.merfish_config.all_codeword_vectors.astype(float))
        self.build_vectors()
        self.updated_measured_barcodes = self.measured_barcodes
        # self.update_binary()



        """ Remove Barcodes without atleast 3 bits"""
        mask = self.barcodes.sum(1)<3
        if mask.numpy().sum()>0:
            self.update_user("Removing Barcodes without 3 1's that were measured") 
            self.barcodes[mask,:] = 1
            # self.barcodes = self.barcodes[mask,:]
            # self.merfish_config.aids = list(np.array(self.merfish_config.aids)[mask.numpy()])

        values,cwords = torch.cdist(self.updated_measured_barcodes.float(),self.barcodes.float()).min(1)
        values = values**2 # Return to bitwise distance
        values = torch.tensor([round(i) for i in values.numpy()])
        """ Filter to good decoded """
        mask = values<2 # CHANGE 2 bit error max
        self.good_values = values[mask]
        self.good_cwords = cwords[mask]
        self.good_indices = torch.tensor(np.array(range(self.updated_measured_barcodes.shape[0])))[mask]
        self.updated_measured_barcodes = self.updated_measured_barcodes[mask,:]
        if self.verbose:
            self.update_user(str(self.good_indices.shape[0])+' Transcripts Found')
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
            self.og_spots = self.spots.copy()
        
    def collapse_spots(self):
        """ Generate transcript dataframe"""
        if self.verbose:
            self.update_user('Collapsing Spots')
        """ Collapse Spots to Counts """
        transcripts = []
        if self.verbose:
            iterable = tqdm(enumerate(list(self.good_indices.numpy())),total=len(list(self.good_indices.numpy())))
        else:
            iterable = enumerate(list(self.good_indices.numpy()))
        for i,idx in iterable:
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
            """ Vector of Signal?"""
            V = self.stk[int(temp.y.mean()),int(temp.x.mean()),:] ### Pull positive bits from called spots 
            for spot,row in temp.iterrows():
                V[int(row['bit'])] = self.stk[int(row.y),int(row.x),int(row['bit'])]
            B = self.barcodes[cword_idx]
            NV = np.sqrt((V**2)/(V**2).sum())
            NB = np.sqrt((B.numpy()**2)/(B.numpy()**2).sum())
            cdist = np.sqrt(np.sum((NB-NV)**2))
            temp = temp.mean()
            temp['intensity'] = temp['signal']
            temp['dispersion'] = dispersion
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
            temp['cdist'] = cdist
            transcripts.append(pd.DataFrame(temp).T)
        transcripts = pd.concat(transcripts,ignore_index=True)
        transcripts = transcripts.dropna()
        transcripts = transcripts.drop(columns=['bit'])
        transcripts['cword_distance'] = [int(round(float(i))) for i in transcripts['cword_distance']]
        transcripts['cword_idx'] = [int(round(float(i))) for i in transcripts['cword_idx']]
        transcripts['gene'] = np.array(self.merfish_config.aids)[transcripts.cword_idx]
        self.transcripts = transcripts 
        
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
        self.update_user('Assigning Cells')
        self.load_segmentation()
        pixel_x = np.array(self.transcripts.x).astype(int)
        pixel_y = np.array(self.transcripts.y).astype(int)
        cell_labels = self.cytoplasm_mask[pixel_y,pixel_x]
        nuclei_labels = self.nuclei_mask[pixel_y,pixel_x]
        self.transcripts['posname'] = self.posname
        self.transcripts['cell_label'] = cell_labels
        self.transcripts['nuclei_label'] = nuclei_labels
            
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


    def predict_real_spots(self,training_data,application_data,columns = ['mass','intensity','size','ecc','ep','X'],label='X'):
        if self.verbose:
            self.update_user('Training Logistic')
        if label == 'cword_idx':
            converter = {i:'True' for i in training_data['cword_idx'].unique()}
            blank_indices = np.array([i for i,gene in enumerate(self.merfish_config.aids) if 'lank' in gene])
            for i in blank_indices:
                converter[i] = 'False'
            training_data['X'] = [converter[i] for i in training_data['cword_idx']]
        training_data['X'] = training_data['X'].astype(str)
        from sklearn.model_selection import train_test_split
        data = training_data[columns]
        data_true = data.loc[data[data['X'].astype(str)=='True'].index]
        data_false = data.loc[data[data['X'].astype(str)=='False'].index]
        if data_false.shape[0]==0:
            application_data['predicted_X'] = 'True'
            application_data['probabilities_X'] = 0
        elif data_true.shape[0]==0:
            application_data['predicted_X'] = 'False'
            application_data['probabilities_X'] = 0 
        else:
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
            if self.verbose:
                print(classification_report(y_test,predictions))

            """ Apply to application"""
            if self.verbose:
                self.update_user('Applying Logistic')
            data = application_data[[i for i in columns if not 'X'==i]]
            X = self.scaler.transform(data)
            application_data['predicted_X'] = self.logmodel.predict(X)
            application_data['probabilities_X'] = self.logmodel.predict_proba(X)[:,1] 

        return application_data

    def logistic_filter(self):
        """ First Use full transcripts to predict real vs false transcripts using false positives"""
        if self.verbose:
            self.check_false_positive_rate(self.transcripts)
        self.transcripts = self.predict_real_spots(self.transcripts,self.transcripts,columns = ['raw_mass', 'ep','intensity', 'signal', 'noise', 'signal-noise','X'],label='cword_idx')
        if self.verbose:
            self.check_false_positive_rate(self.transcripts[self.transcripts['X']=='True'])
        """ Use that prediction to train and predict real vs false spots"""
        self.spots = self.predict_real_spots(self.transcripts,self.og_spots,columns = ['mass','intensity','size','ecc','ep','X'],label='cword_idx')
        if self.verbose:
            self.update_user('Before Filter '+str(self.spots.shape[0]))
        self.spots = self.spots[self.spots['X']=='True']
        if self.verbose:
            self.update_user('After Filter '+str(self.spots.shape[0]))
        self.spots = self.spots.drop(columns=['X'])

    def check_false_positive_rate(self,transcripts):
        blank_indexes = np.array([i for i,gene in enumerate(np.array(self.merfish_config.aids)) if 'lank' in gene])
        correction = np.array(self.merfish_config.aids).shape[0]/blank_indexes.shape[0]
        n_false = np.sum(np.isin(transcripts.cword_idx,blank_indexes))
        fpr = correction*n_false/transcripts.shape[0]
        self.update_user('False Positive Rate: '+str(int(10000*fpr)/(10000/100))+'%')
        self.update_user(str(n_false) + ' False Positives')
        self.update_user(str(transcripts.shape[0]) + ' All Positives')

        

from scipy.spatial import cKDTree
def MNN_Agglomerative(X,max_distance = 20,max_iterations=10,verbose = False):
    cluster_df = pd.DataFrame(index=range(X.shape[0]))
    dims = list(np.array(range(X.shape[1])))
    for dim in dims:
        cluster_df[dim] = X[:,dim]
        cluster_df[str(dim)+'_std'] = 0
    cluster_df['idxes'] = [[i] for i in range(X.shape[0])]
    cluster_df['labels'] = 0
    og_cluster_df = cluster_df.copy()
    for iteration in range(max_iterations):
        cluster_X = np.array(cluster_df[dims])
        Xindex = cKDTree(cluster_X)
        X_dists, X_idx = Xindex.query(cluster_X, k=2)
        mnn = (X_idx[X_idx[:,1],1]==X_idx[:,0])
        mnn = mnn&(X_dists[:,1]<max_distance)
        points = X_idx[mnn,0]
        seeds = list(np.unique(X_idx[points,:].min(1)))
        # for idx in points:
        #     if not X_idx[idx,1] in seeds:
        #         seeds.append(idx)
        if len(seeds)==0:
            break
        new_cluster_X = np.zeros([len(seeds),len(dims)])
        new_cluster_idxes = []
        new_cluster_labels = np.zeros(len(seeds))
        new_cluster = {}
        for dim in dims:
            new_cluster[dim] = np.zeros(len(seeds))
            new_cluster[str(dim)+'_std'] = np.zeros(len(seeds))
        if verbose:
            iterable = tqdm(enumerate(seeds),total=len(seeds))
        else:
            iterable = enumerate(seeds)
        for i,seed in iterable:
            if new_cluster_labels.max()==0:
                new_cluster_label = cluster_df['labels'].max()+1
            else:
                new_cluster_label = new_cluster_labels.max()+1
            pair = [seed,X_idx[seed,1]]
            temp = cluster_df['idxes'].iloc[pair[0]]
            temp.extend(cluster_df['idxes'].iloc[pair[1]])
            temp = list(np.unique(temp))
            new_cluster_idxes.append(temp)
            for dim in dims:
                new_cluster[dim][i] = og_cluster_df[dim].iloc[temp].mean()
                new_cluster[str(dim)+'_std'][i] = og_cluster_df[dim].iloc[temp].std()
            new_cluster_labels[i] = new_cluster_label
        new_cluster_df = pd.DataFrame(index=range(len(seeds)))
        for dim in dims:
            new_cluster_df[dim] = new_cluster[dim]
            new_cluster_df[str(dim)+'_std'] = new_cluster[str(dim)+'_std']
        new_cluster_df['idxes'] = new_cluster_idxes
        new_cluster_df['labels'] = new_cluster_labels
        points_indexes = cluster_df.iloc[points].index
        cluster_df = cluster_df.drop(index = points_indexes)
        cluster_df = pd.concat([cluster_df,new_cluster_df],ignore_index=True)
        if max_iterations==iteration:
            print('Failed To Coverge')
    cluster_df = cluster_df.drop_duplicates(subset=dims)
    cluster_df['n_spots'] = [len(i) for i in cluster_df['idxes']]
    X_labels = np.zeros(X.shape[0])
    for idx,row in cluster_df[cluster_df['labels']!=0].iterrows():
        X_labels[row['idxes']] = row['labels']
    return X_labels,cluster_df