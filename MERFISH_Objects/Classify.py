from tqdm import tqdm
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

class Classify_Class(object):
    def __init__(self,
                 metadata_path,
                 dataset,
                 posname,
                 cword_config,
                 wait=300,
                 verbose=False,
                 hybedata_path=''):
        self.metadata_path = metadata_path
        self.dataset = dataset
        self.posname = posname
        self.verbose = verbose
        self.wait = wait
        self.hybedata_path = hybedata_path
        if len(self.hybedata_path)>0:
            self.hybedata = HybeData(self.hybedata_path)
        self.cword_config = cword_config
        self.merfish_config = importlib.import_module(self.cword_config)
        self.parameters =  self.merfish_config.parameters
        self.utilities = Utilities_Class(self.parameters['utilities_path'])
        self.fishdata = FISHData(os.path.join(self.metadata_path,self.parameters['fishdata']))
        self.two_dimensional = self.parameters['two_dimensional']
        self.completed = False
        self.passed = True
     
    def run(self):
        self.check_flags()

    def classify(self):
        self.load_configuration()
        self.generate_vectors()
        if self.parameters['classification_method']=='Logistic':
            self.initialize_snr_thresh()
            self.binarize_vectors()
            self.fit_models()
            self.call_bits()
            self.fit_bitmatch()
        elif self.parameters['classification_method']=='Euclidean':
            self.euclidean_classify_pixels()
        else:
            raise('Unknown classification_method'+self.parameters['classification_method'])
        self.parse_classification()
        self.load_masks()
        self.assign_to_cells()
        self.generate_cell_metadata()
        self.generate_counts()
        self.save_data()
            
    def check_flags(self):
        self.failed = False
        if self.verbose:
            i = [i for i in tqdm([],desc='Checking Flags')]
        # Classification
        if not self.failed:
            flag = self.fishdata.load_data('flag',dataset=self.dataset,
                                           posname=self.posname,hybe='all')
            if flag == 'Failed':
                log = ''
                self.failed = True
                self.completed = True
            if flag == 'Passed':
                self.completed = True
        # check if counts exist
        counts = self.fishdata.load_data('counts',dataset=self.dataset,posname=self.posname)
        if not isinstance(counts,type(None)):
            self.completed = True
            self.fishdata.add_and_save_data('Passed','flag',
                                            dataset=self.dataset,
                                            posname=self.posname,
                                            hybe='all')
        # Position
        if not self.failed:
            flag = self.fishdata.load_data('flag',dataset=self.dataset,posname=self.posname)
            if flag == 'Failed': 
                log = self.posname+' Failed'
                self.failed = True
                self.completed = True
        # Segmentation
        if not self.failed:
            flag = self.fishdata.load_data('flag',dataset=self.dataset,
                                           posname=self.posname,
                                           channel=self.parameters['nucstain_channel'])
            if flag == 'Failed': 
                log = 'Segmentation Failed'
                self.failed = True
                self.completed = True
                
        if self.failed:
            self.fishdata.add_and_save_data('Failed','flag',
                                            dataset=self.dataset,
                                            posname=self.posname,
                                            hybe='all')
            if len(log)>0:
                self.fishdata.add_and_save_data(log,'log',
                                                dataset=self.dataset,
                                                posname=self.posname,
                                                hybe='all')
        
        if not self.failed:
            if not self.completed:
                self.classify()

    def check_projection(self):
        self.projection_zstart=self.parameters['projection_zstart'] 
        self.projection_k=self.parameters['projection_k']
        self.projection_zskip=self.parameters['projection_zskip'] 
        self.projection_zend=self.parameters['projection_zend']
        self.projection_function=self.parameters['projection_function']
        if self.verbose:
            i = [i for i in tqdm([],desc='Checking Projection Zindexes')]
        acq = [i for i in os.listdir(self.metadata_path) if 'hybe1' in i][0]
        self.image_table = pd.read_csv(os.path.join(self.metadata_path,acq,'Metadata.txt'),sep='\t')
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
        elif zend<zstart:
            print('zstart of ',self.projection_zstart,' is larger than zend of', self.projection_zend)
            raise(ValueError('Projection Error'))
        self.zindexes = np.array(range(self.projection_zstart,self.projection_zend,self.projection_zskip))
        if self.parameters['two_dimensional']:
            self.zindexes = [0]
        
    def load_configuration(self):
        self.merfish_config = importlib.import_module(self.cword_config)
        self.bitmap = self.merfish_config.bitmap
        self.genes = self.merfish_config.gids + self.merfish_config.bids
        self.barcodes = self.merfish_config.all_codeword_vectors
        self.barcodes = torch.FloatTensor(self.barcodes*2 - 1)
        self.nbits = self.merfish_config.nbits
        self.channels = [channel for seq,hybe,channel in self.bitmap]
        self.hybes = [hybe for seq,hybe,channel in self.bitmap]
        self.blank_indexes = [i for i,gene in enumerate(self.genes) if 'blank' in gene]
        
        self.parameters = self.merfish_config.parameters
        self.utilities_path = self.parameters['utilities_path']
        self.fishdata = FISHData(os.path.join(self.metadata_path,self.parameters['fishdata']))
        self.check_projection()
        self.load_segmentation()
        self.bitmatch_thresh = self.parameters['match_thresh']
        self.fpr_thresh = self.parameters['fpr_thresh']
        
    def load_segmentation(self):
        self.segmentation = torch.zeros([2048,2048,len(self.zindexes)])
        if self.parameters['segment_two_dimensional']:
            mask = self.fishdata.load_data('cytoplasm_mask',dataset=self.dataset,posname=self.posname)
            if not isinstance(mask,type(None)):
                self.segmentation[:,:,0] = torch.tensor(mask.astype(float))
        else:
            if self.verbose:
                iterable = tqdm(enumerate(self.zindexes),total=len(self.zindexes),desc='Loading Segmentation')
            else:
                iterable = enumerate(self.zindexes)
            for i,z in iterable:
                mask = self.fishdata.load_data('cytoplasm_mask',dataset=self.dataset,posname=self.posname,zindex=z)
                if not isinstance(mask,type(None)):
                    self.segmentation[:,:,i] = torch.tensor(mask.astype(float))
        self.seg_mask = self.segmentation.max(axis=2).values>0
        if (self.seg_mask).sum()==0:
            if self.verbose:
                self.seg_mask = self.seg_mask==False
                print('No Cells in '+self.posname)
                """ Fail Position"""
        
    def load_codestack(self,zindex):
        if len(self.hybedata_path)>0:
            cstk = self.hybedata.load_data(self.posname,zindex,'cstk')
            if isinstance(cstk,type(None)):
                cstk = torch.zeros((2048,2048,self.nbits))
            else:
                cstk = torch.FloatTensor(cstk)
        else:
            cstk = torch.zeros((2048,2048,self.nbits))
            for bitmap_idx in range(self.nbits):
                seq,hybe,channel = self.bitmap[bitmap_idx]
                temp = self.fishdata.load_data('image',dataset=self.dataset,posname=self.posname,hybe=hybe,channel=channel,zindex=zindex)
                if not isinstance(temp,type(None)):
                    cstk[:,:,bitmap_idx] = torch.tensor(temp.astype('float32'),dtype=torch.float32)
        return cstk
    
    def pull_vector_zindex(self,zindex):
        cstk = self.load_codestack(zindex)
        
        cstk[self.seg_mask==False,:] = 0
        cstk = cstk-cstk.median(axis=2).values[:,:,None]
        cstk = self.z_score_cstk(cstk)
        self.cstk = cstk
        if torch.max(cstk)>0:
            cstk_max = torch.max(cstk,axis=2).values
            # torch percentile should be faster
            mask = cstk_max>np.percentile(cstk_max[self.seg_mask].cpu(),90) 
            # Add later self.parameters.classification_rel_thresh)
            x,y = torch.where(mask)
            v = cstk[x,y,:].cpu()
            z = zindex*torch.ones(len(x),dtype=int)
            return v,x,y,z
        else:
            return [],[],[],[]
        
    def generate_vectors(self):
        V = []
        X = []
        Y = []
        Z = []
        if self.verbose:
            iterable = tqdm(self.zindexes,desc='Generating Vectors')
        else:
            iterable = self.zindexes
        for zindex in iterable:
            v,x,y,z = self.pull_vector_zindex(zindex)
            if len(v)>0:
                V.append(v)
                X.append(x)
                Y.append(y)
                Z.append(z)
                
        self.vectors = torch.cat(V)
        self.x_coordinates = torch.cat(X)
        self.y_coordinates = torch.cat(Y)
        self.z_coordinates = torch.cat(Z)
        
    def z_score_cstk(self,cstk):
        mu = torch.tensor([torch.mean(cstk[:,:,i][self.seg_mask]) for i in range(self.nbits)])
        std = torch.tensor([torch.std(cstk[:,:,i][self.seg_mask]) for i in range(self.nbits)])
        return (cstk-mu)/std
    
    def calculate_false_positive_rate(self,pixel_labels):
        return (len(self.genes)/len(self.blank_indexes))*(np.sum(np.isin(np.array(pixel_labels),self.blank_indexes))/len(pixel_labels))
    
    def initialize_snr_thresh(self):
        res = []
        if self.bitmatch_thresh<0:
            self.bitmatch_thresh = self.nbits+self.bitmatch_thresh
        # Move to parameters
        if self.verbose:
            iterable = tqdm(range(5,15),desc='Initalizing S/N Threshold')
        else:
            iterable = range(5,15)
        for snr_thrsh in iterable: # move to parameters
            bit_calls = (self.vectors>snr_thrsh).float()*2 - 1 # Converts to + and -
            bit_match = bit_calls.mm(self.barcodes.t())
            init_pixel_indexes,init_pixel_labels = (bit_match>=self.nbits-2).nonzero(as_tuple=True)
            fpr = self.calculate_false_positive_rate(init_pixel_labels)                
            res.append((snr_thrsh, self.bitmatch_thresh, fpr, len(init_pixel_labels), len(init_pixel_labels)/len(init_pixel_labels.unique())))
            if fpr<(2*self.fpr_thresh):
                self.snr_thrsh = snr_thrsh
                break
        self.res = pd.DataFrame(res).sort_values(2,ascending=False)
        self.filtered_res = self.res[self.res[2]<(2*self.fpr_thresh)]
        if len(self.filtered_res)==0:
            self.snr_thrsh = self.res[0].max()
            if self.verbose:
                print('False Positive Rate of'+str(round(100*self.res[2].min(),2))+'% > '+str(200*self.fpr_thresh)+'%')
                print('Try Higher S/N Thresholds in future')
        else:
            self.snr_thrsh = self.filtered_res[0].iloc[0]
            
    def binarize_vectors(self):
        bit_calls = (self.vectors>self.snr_thrsh).float()*2 - 1
        bit_match = bit_calls.mm(self.barcodes.t())
        self.init_pixel_indexes,self.init_pixel_labels = (bit_match>=(self.nbits-2)).nonzero(as_tuple=True)
        
    def euclidean_classify_pixels(self):
        mu = self.vectors.mean(1)
        std = self.vectors.std(1)
        zscored = (self.vectors-mu)/std
        norm_vectors = torch.tensor(normalize(zscored))
        norm_barcodes = torch.tensor(normalize(self.barcodes))
        cdist = torch.cdist(norm_vectors,norm_barcodes)
        cdist_min,cdist_index = cdist.min(1)
        # Set a threshold
        thresh = 0.52 # a 2 bit error
        mask = cdist_min>thresh
        false_mask = torch.tensor(np.isin(cdist_index[mask],self.blank_indexes))
        correction_bitwise = self.barcodes.shape[0]/self.blank_indexes.shape[0]
        true_mask = false_mask==False
        self.false_calls = self.barcodes[cdist_index[mask][false_mask],:].sum(0)
        self.total_calls = self.barcodes[cdist_index[mask],:].sum(0)
        self.fpr_bitwise = correction_bitwise*self.false_calls/self.total_calls
        self.pixel_indexes = torch.where(mask)
        self.pixel_labels = cdist_index[mask]
        
    def fit_models(self):
        models = {}
        if self.verbose:
            iterable = tqdm(range(self.nbits),desc='Fitting Bit Models')
        else:
            iterable = range(self.nbits)
        for i in iterable:
            x = self.vectors[self.init_pixel_indexes,i:i+1]
            y = self.barcodes[self.init_pixel_labels,i]
            model = LogisticRegression(solver='lbfgs')
            model.fit(x,y)
            models[i] = model
        self.models = models
        
    def qc_models(self):
        """ Validate that the models are reasonable"""
        print('Not Implemented')
        
    def save_models(self):
        if self.verbose:
            i = [i for i in tqdm([],desc='Saving Models')]
        self.utilities = Utilities_Class(self.utilities_path)
        self.utilities.save_data(self.models,Dataset=self.dataset,Type='models')
        self.utilities.save_data(self.bitmatch_thresh,Dataset=self.dataset,Type='bitmatch_thresh')
        self.utilities.save_data(self.snr_thrsh,Dataset=self.dataset,Type='snr_thresh')
        
    def load_models(self):
        if self.verbose:
            i = [i for i in tqdm([],desc='Loading Models')]
        self.utilities = Utilities_Class(self.utilities_path)
        self.models = self.utilities.load_data(Dataset=self.dataset,Type='models')
        self.bitmatch_thresh = self.utilities.load_data(Dataset=self.dataset,Type='bitmatch_thresh')
        self.snr_thrsh = self.utilities.load_data(Dataset=self.dataset,Type='snr_thrsh')
    
    def call_bits(self):
        bit_calls = []
        if self.verbose:
            iterable = tqdm(range(self.nbits),desc='Classifying Pixels')
        else:
            iterable = range(self.nbits)
        for i in iterable:
            x = self.vectors[:,i:i+1]
            bit_calls.append(torch.FloatTensor(self.models[i].predict_proba(x)[:,1]))
        bit_calls = torch.stack(bit_calls, 1)
        self.bit_match = (bit_calls*2-1).mm((self.barcodes).t())
        self.bitmatch_thresh = self.nbits-6
        self.pixel_indexes,self.pixel_labels = (self.bit_match>=self.nbits-6).nonzero(as_tuple=True)
        self.bit_match_values = self.bit_match.max(axis=1).values[self.pixel_indexes]

    def fit_bitmatch(self):
        bitmatch_threshes = np.flipud(np.linspace(self.nbits-6,self.nbits,num=100))
        if self.verbose:
            iterable = tqdm(enumerate(bitmatch_threshes),total=len(bitmatch_threshes),desc='Fit Bitmatch Threshold')
        else:
            iterable = enumerate(bitmatch_threshes)
        for i,bitmatch_thresh in iterable:
            labels = self.pixel_labels[self.bit_match_values>bitmatch_thresh]
            if len(labels)>1000:
                fpr = self.calculate_false_positive_rate(labels)
                if fpr>self.fpr_thresh:
                    self.bitmatch_thresh = bitmatch_threshes[i-1]
                    self.pixel_indexes,self.pixel_labels = (self.bit_match>=self.bitmatch_thresh).nonzero(as_tuple=True)
                    self.bit_match_values = self.bit_match.max(axis=1).values[self.pixel_indexes]
                    break
                    
    def fit_bitmatch_spots(self):
        bitmatch_threshes = np.flipud(np.linspace(self.spots.bitmatch.min(),self.spots.bitmatch.max(),num=100))
        if self.verbose:
            iterable = tqdm(enumerate(bitmatch_threshes),total=len(bitmatch_threshes),desc='Fit Bitmatch Threshold')
        else:
            iterable = enumerate(bitmatch_threshes)
        master_labels =self.spots.cword_idx 
        bitmatch = self.spots.bitmatch
        for i,bitmatch_thresh in iterable:
            labels = master_labels[bitmatch>bitmatch_thresh]
            if len(labels)>1000:
                fpr = self.calculate_false_positive_rate(labels)
                if fpr>self.fpr_thresh:
                    self.bitmatch_thresh = bitmatch_threshes[i-1]
                    break
        self.spots = self.spots[self.spots.bitmatch>self.bitmatch_thresh]
        
    
    def parse_classification(self):
        self.pixel_indexes = torch.LongTensor(self.pixel_indexes)
        x = self.x_coordinates[self.pixel_indexes].type(torch.LongTensor)
        y = self.y_coordinates[self.pixel_indexes].type(torch.LongTensor)
        z = self.z_coordinates[self.pixel_indexes].type(torch.LongTensor)
        classification_stack = -1*torch.ones([2048,2048,int(z.max())+1])
        classification_stack[x,y,z] = self.pixel_labels.float()
        bitmatch_stack = -1*torch.ones([2048,2048,int(z.max())+1])
        bitmatch_stack[x,y,z] = self.bit_match[self.pixel_indexes,self.pixel_labels].float()
        projected_z = z.unique()
        projected_stack = classification_stack[:,:,projected_z]
        projected_labels = label(np.array(projected_stack),connectivity=2)
        labels = torch.zeros_like(classification_stack,dtype=int)
        labels[:,:,projected_z] = torch.tensor(projected_labels)
        label_vector = labels[x,y,z]
        barcodes = 1*(self.barcodes>0)
        intensities = self.vectors[self.pixel_indexes,:]*barcodes[self.pixel_labels]
        intensities[intensities==0] = np.nan
        mean_intensity = torch.tensor(np.nanmean(intensities,axis=1))
        sum_intensity = torch.tensor(np.nansum(intensities,axis=1))
        cword_idxes = self.pixel_labels
        gene_call_rows = []
        x = x.float()
        y = y.float()
        z = z.float()
        if self.verbose:
            iterable = tqdm(torch.unique(label_vector),desc='Parse Classification')
        else:
            iterable = torch.unique(label_vector)
        for idx in iterable:
            mask = label_vector==idx
            loc = torch.where(mask)
            cword_idx = cword_idxes[loc][0]
            if cword_idx ==-1:
                continue
            n_pixels = len(loc[0])
            if n_pixels>20:
                continue
            tx = x[loc].float()
            ty = y[loc].float()
            tz = z[loc].float()
            c_x = torch.mean(tx)
            c_y = torch.mean(ty)
            c_z = torch.median(tz)
            spot_sum = torch.sum(sum_intensity[loc])
            spot_ave = torch.mean(mean_intensity[loc])
            gene = self.genes[cword_idx]
            bitmatch = torch.mean(bitmatch_stack[tx.long(),ty.long(),tz.long()]) # maybe max?
    #         spot_distance = torch.mean(cword_distances[loc]) # maybe min?
            gene_call_rows.append([gene, float(spot_sum), float(c_x),float(c_y),float(c_z),
                                float(spot_ave), int(n_pixels), int(cword_idx),float(bitmatch)])
        self.spots = pd.DataFrame(gene_call_rows, columns=['gene', 'ssum', 
                                                   'pixel_x','pixel_y',
                                                   'pixel_z', 'ave', 
                                                   'npixels', 'cword_idx','bitmatch'])
#         self.fit_bitmatch_spots()
        self.false_positive_rate = self.calculate_false_positive_rate(self.spots.cword_idx)        
        
    def assign_to_cells(self):
        pixel_size = 0.103
        pixel_z_converter = {z:i for i,z in enumerate(self.zindexes)}
        self.posname_x,self.posname_y = self.pos_metadata.XY.iloc[0]
        pixel_x = np.array(self.spots.pixel_x).astype(int)
        pixel_y = np.array(self.spots.pixel_y).astype(int)
        pixel_z = np.array(self.spots.pixel_z).astype(int)
        pixel_z = np.array([pixel_z_converter[z] for z in pixel_z]).astype(int)
        stage_x = pixel_x*pixel_size+self.posname_x
        stage_y = pixel_y*pixel_size+self.posname_y
        cell_labels = self.cytoplasm_mask[pixel_x,pixel_y,pixel_z]
        nuclei_labels = self.nuclei_mask[pixel_x,pixel_y,pixel_z]
        self.spots['posname'] = self.posname
        self.spots['cell_label'] = cell_labels
        self.spots['nuclei_label'] = nuclei_labels
        self.spots['stage_x'] = stage_x
        self.spots['stage_y'] = stage_y
        
        
    def load_masks(self):
        """ Implement """
        self.check_projection()
        if self.verbose:
            iterable = tqdm(enumerate(self.zindexes),total=len(self.zindexes),desc='Loading Masks')
        else:
            iterable = enumerate(self.zindexes)
        nuclei_mask = np.zeros([2048,2048,len(self.zindexes)])
        cytoplasm_mask = np.zeros([2048,2048,len(self.zindexes)])
        if self.parameters['segment_two_dimensional']:
            nuclei_mask_2d = self.fishdata.load_data('nuclei_mask',
                                                        dataset=self.dataset,
                                                        posname=self.posname)
            cytoplasm_mask_2d = self.fishdata.load_data('cytoplasm_mask',
                                                        dataset=self.dataset,
                                                        posname=self.posname)
            for i,z in iterable:
                nuclei_mask[:,:,i] = nuclei_mask_2d
                cytoplasm_mask[:,:,i] = cytoplasm_mask_2d
        else:
            for i,z in iterable:
                nuclei_mask[:,:,i] = self.fishdata.load_data('nuclei_mask',
                                                            dataset=self.dataset,
                                                            posname=self.posname,
                                                            zindex=z)
                cytoplasm_mask[:,:,i] = self.fishdata.load_data('cytoplasm_mask',
                                                            dataset=self.dataset,
                                                            posname=self.posname,
                                                            zindex=z)
        self.nuclei_mask = nuclei_mask
        self.cytoplasm_mask = cytoplasm_mask
        if np.sum(self.nuclei_mask)==0:
            # There are no cells
            if self.verbose:
                print('No Cells for '+self.posname)
            
        
    def check_projection(self):
        self.projection_zstart=self.parameters['projection_zstart'] 
        self.projection_k=self.parameters['projection_k']
        self.projection_zskip=self.parameters['projection_zskip'] 
        self.projection_zend=self.parameters['projection_zend']
        self.projection_function=self.parameters['projection_function']
        if self.verbose:
            i = [i for i in tqdm([],desc='Checking Projection Zindexes')]
        # allows faster loading of metadata
        self.acq = [i for i in os.listdir(self.metadata_path) if 'nucstain' in i][0]
        self.metadata = Metadata(os.path.join(self.metadata_path,self.acq))
        self.pos_metadata = self.metadata.image_table[(self.metadata.image_table.Position==self.posname)]
        self.len_z = len(self.pos_metadata.Zindex.unique())
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
        elif zend<zstart:
            print('zstart of ',self.projection_zstart,' is larger than zend of', self.projection_zend)
            raise(ValueError('Projection Error'))
        self.zindexes = np.array(range(self.projection_zstart,self.projection_zend,self.projection_zskip))
        self.nZ = len(self.zindexes)
        if self.parameters['two_dimensional']:
            self.zindexes = [0]
    
    def generate_cell_metadata(self,pixel_size=0.104,z_step=0.4):
        columns = ['posname','label','pixel_x','pixel_y','pixel_z',
                   'voronoi_volume','nuclei_volume','voronoi_nz',
                   'nuclei_nz','stage_x','stage_y','stage_z',
                   'voronoi_ntranscripts','nuclei_ntranscripts']
        voronoi_regions = regionprops(self.cytoplasm_mask.astype(int))
        nuclei_regions = regionprops(self.nuclei_mask.astype(int))
        cell_data = []
        if self.verbose:
            iterable = tqdm(range(len(nuclei_regions)))
        else:
            iterable = range(len(nuclei_regions))
        for idx in iterable:
            voronoi_region = voronoi_regions[idx]
            nuclei_region = nuclei_regions[idx]
            label = nuclei_region.label
            pixel_x,pixel_y,pixel_z = nuclei_region.centroid
            voronoi_volume = voronoi_region.coords.shape[0]*pixel_size*pixel_size*z_step
            nuclei_volume = nuclei_region.coords.shape[0]*pixel_size*pixel_size*z_step
            voronoi_nz = len(np.unique(voronoi_region.coords[:,2]))
            nuclei_nz = len(np.unique(nuclei_region.coords[:,2]))
            stage_x = self.posname_x+(pixel_x*pixel_size)
            stage_y = self.posname_y+(pixel_y*pixel_size)
            stage_z = z_step*pixel_z
            voronoi_ntranscripts = np.sum(self.spots.cell_label==label)
            nuclei_ntranscripts = np.sum(self.spots.nuclei_label==label)
            out = [self.posname,label,pixel_x,pixel_y,pixel_z,
                   voronoi_volume,nuclei_volume,voronoi_nz,
                   nuclei_nz,stage_x,stage_y,stage_z,
                   voronoi_ntranscripts,nuclei_ntranscripts]
            cell_data.append(out)
        self.cell_metadata = pd.DataFrame(cell_data,columns=columns)

    def generate_counts(self):
        cells = self.spots.cell_label.unique()
        counts = np.zeros([len(cells),len(self.genes)],dtype=int)
        if self.verbose:
            iterable = tqdm(enumerate(cells),total=len(cells),desc='Generating Counts')
        else:
            iterable = enumerate(cells)
        for i,cell in iterable:
            for cword_idx,cc in Counter(self.spots[self.spots.cell_label==cell].cword_idx).items():
                counts[i,cword_idx] = cc
        self.counts = pd.DataFrame(counts,columns=self.genes,index=cells)
        
    def save_data(self):
        self.fishdata.add_and_save_data(self.counts,
                                        'counts',
                                        dataset=self.dataset,
                                        posname=self.posname)
        self.fishdata.add_and_save_data(self.cell_metadata,
                                        'cell_metadata',
                                        dataset=self.dataset,
                                        posname=self.posname)        
        self.fishdata.add_and_save_data(self.spots,
                                        'spotcalls',
                                        dataset=self.dataset,
                                        posname=self.posname)
        self.fishdata.add_and_save_data('Passed','flag',
                                        dataset=self.dataset,
                                        posname=self.posname,
                                        hybe='all')
        self.completed = True