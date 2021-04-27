import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metadata import Metadata
import os
from fish_results import HybeData
from collections import defaultdict, Counter
from scipy import stats
import seaborn as sns
from scipy.stats import spearmanr
from skimage.transform import resize
from skimage.measure import regionprops
from math import sqrt
import importlib
import multiprocessing
import sys
# plt.style.use(['dark_background'])
# from ipypb import ipb
from tqdm import tqdm
import random

def ReadsPerGene_fun(system='cornea'):
    if system=='cornea':
        f = '/bigstore/GeneralStorage/Zach/Cornea_RNAseq/Aligned/ReadsPerGene.xlsx'
        ReadsPerGene = pd.read_excel(f)
        ReadsPerGene.index = ReadsPerGene.GeneIDs
    elif system=='3t3':
        f = '/bigstore/GeneralStorage/Evan/NFKB_MERFISH/Calibration_Set_Data/3T3_Calibration_Set/RNA_Seq_Data/Hoffmann_IFNAR_KO_3T3_TNF.txt'
        ReadsPerGene = pd.read_csv(f,sep='\t')
        gids = []
        for gid in ReadsPerGene.Geneid:
            gids.append(gid.split('.')[0])
        ReadsPerGene.Geneid = gids
        ReadsPerGene.index = gids
    else:
        print('Unknown System')
    return ReadsPerGene

def merfish_correlation(spotcalls,color='b',alpha=1,label='spotcalls',system='cornea',ReadsPerGene=False):
    import pandas as pd
    if not isinstance(ReadsPerGene,pd.core.frame.DataFrame):
        if system=='cornea':
            f = '/bigstore/GeneralStorage/Zach/Cornea_RNAseq/Aligned/ReadsPerGene.xlsx'
            ReadsPerGene = pd.read_excel(f)
            ReadsPerGene.index = ReadsPerGene.GeneIDs
        elif system=='3t3':
            f = '/bigstore/GeneralStorage/Evan/NFKB_MERFISH/Calibration_Set_Data/3T3_Calibration_Set/RNA_Seq_Data/Hoffmann_IFNAR_KO_3T3_TNF.txt'
            ReadsPerGene = pd.read_csv(f,sep='\t')
            gids = []
            for gid in ReadsPerGene.Geneid:
                gids.append(gid.split('.')[0])
            ReadsPerGene.Geneid = gids
            ReadsPerGene.index = gids
        else:
            print('Unknown System')
    GeneList = pd.read_csv('/bigstore/GeneralStorage/Zach/MERFISH/Inflammatory/InflammationGeneList.csv')
    GeneList.index = GeneList.Gene
    counts = []
    fpkms = []
    from collections import defaultdict, Counter
    FISH_Spots = Counter(spotcalls.gene)
    for gn,cc in FISH_Spots.items():
        if 'blank' in gn:
            continue
        else:
            gid = GeneList.loc[gn]['Gene_ID']
            if system=='cornea':
                reads = ReadsPerGene.loc[gid].Unstranded
            elif system=='3t3':
                reads = ReadsPerGene.loc[gid]['IFNAR-TNF-1_tot.bam']
            else:
                print('Unknown System')
            fpkm = reads/GeneList.loc[gn]['Length']
            if isinstance(fpkm,np.float64):
                if cc<2:
                    continue
                counts.append(cc)
                fpkms.append(fpkm)
    from scipy.stats import spearmanr
    import matplotlib.pyplot as plt
    plt.scatter(np.log10(fpkms),np.log10(counts),c=color,alpha=alpha,label=label)
    print(spearmanr(fpkms,counts))
    plt.suptitle('Untrimmed FPKM vs Spot Count')
    plt.ylabel('log10 MERFISH Spot Count')
    plt.xlabel('log10 RNAseq FPKM')
    plt.legend()

def ptl_hist(spotcalls,bins=1000,colors='rkb',alpha=0.5,column='ave'):
    import matplotlib.pyplot as plt
    plt.hist(np.log10(spotcalls[column]),bins=bins,color=colors[0],alpha=alpha,label='raw')
    plt.hist(np.log10(spotcalls[spotcalls.npixels>1][column]),bins=bins,color=colors[1],alpha=alpha,label='npixels>1')
    plt.hist(np.log10(spotcalls[spotcalls.ssum>2**12][column]),bins=bins,color=colors[2],alpha=alpha,label='ssum>2**12')
    plt.ylabel('Counts')
    plt.xlabel(str('Log10 '+str(column)))
    plt.legend()
    plt.show()
    
def spotcalls_qc(spotcalls,system='cornea',colors='rkb',alpha=0.5,bins=1000):
    import matplotlib.pyplot as plt
    ReadsPerGene=ReadsPerGene_fun(system=system)
    merfish_correlation(spotcalls,color=colors[0],alpha=alpha,label='raw',system=system,ReadsPerGene=ReadsPerGene)
    merfish_correlation(spotcalls[spotcalls.npixels>1],color=colors[1],alpha=alpha,label='npixels>1',system=system,ReadsPerGene=ReadsPerGene)
    merfish_correlation(spotcalls[spotcalls.ssum>2**12],color=colors[2],alpha=alpha,label='ssum>2**12',system=system,ReadsPerGene=ReadsPerGene)
    plt.show()
    ptl_hist(spotcalls,bins=bins,colors=colors,alpha=alpha,column='ave')
    ptl_hist(spotcalls,bins=bins,colors=colors,alpha=alpha,column='ssum')
    
def onfly_qc(md,path=False):
    if path==True:
        from metadata import Metadata
        md = Metadata(md)
    md_path = md.base_pth
    import pickle
    import os
    import time
    i=0
    while i==0:
        try:
            tforms = pickle.load(open(os.path.join(md_path,'results','tforms.pkl'),'rb'))
            beads = pickle.load(open(os.path.join(md_path,'results','beads.pkl'),'rb'))
            i=1
        except:
            print('Waiting')
            time.sleep(5)
    print(len(tforms['good'].keys()),' Good Positions')
    print(len(tforms['bad'].keys()),' Failed Positions')
    import matplotlib.pyplot as plt
    X = []
    Y = []
    good_pos = tforms['good'].keys()
    bad_pos = tforms['bad'].keys()
    for pos in good_pos:
        x,y = md.image_table[md.image_table.Position==pos].XY.iloc[0]
        X.append(x)
        Y.append(y)
    plt.scatter(X,Y,c='g',label='good')
    X = []
    Y = []
    for pos in bad_pos:
        if pos in good_pos:
            continue
        x,y = md.image_table[md.image_table.Position==pos].XY.iloc[0]
        X.append(x)
        Y.append(y)
    plt.scatter(X,Y,c='r',label='bad')
    plt.xlabel('X Stage Coordinate')
    plt.ylabel('Y Stage Coordinate')
    plt.legend()
    plt.show()
    
def photobleach_qc(md,path=True,pos=False):
    import matplotlib.pyplot as plt
    if path==True:
        from metadata import Metadata
        md = Metadata(md)
    if pos ==False:
        pos = md.image_table.Position.iloc[0]
    for acq in md.image_table[md.image_table.Position==pos].acq.unique():
        if 'hybe' in acq:
            stk = md.stkread(Position=pos,Channel='FarRed',acq=acq)
            plt.plot(range(stk.shape[2]),np.mean(np.mean(stk,axis=0),axis=0),label=acq)
    plt.title('FarRed')
    plt.xlabel('Z index')
    plt.ylabel('Average Intensity')
    plt.legend()
    plt.show()
    for acq in md.image_table[md.image_table.Position==pos].acq.unique():
        if 'hybe' in acq:
            stk = md.stkread(Position=pos,Channel='Orange',acq=acq)
            plt.plot(range(stk.shape[2]),np.mean(np.mean(stk,axis=0),axis=0),label=acq)
    plt.title('Orange')
    plt.xlabel('Z index')
    plt.ylabel('Average Intensity')
    plt.legend()
    plt.show()

def img2stage_coordinates(spotcalls,md,pixelsize=0.109,cameradirection=[1,1]):
    """
    RKF Comments - md and path can be combined. if md isinstance(str) then load metadata else 
    assert it isinstance(Metadata) and continue
    
    camera pixels is hardcoded
    
    CoordX and CoordY are not descriptive enough names to know difference between them and 
    other x,y. Candidate suggestion either stageX/Y or globalXY
    """
    if isinstance(md, str):
        md = Metadata(md)
    X = []
    Y = []
    for pos in tqdm(spotcalls.posname.unique()):
        coordX,coordY = md.image_table[md.image_table.Position==pos].XY.iloc[0]
        pos_temp = spotcalls[spotcalls.posname==pos]
        centroids = np.stack(pos_temp.centroid, axis=0)
        centroids = centroids*pixelsize*cameradirection+(coordY, coordX)
        X.extend(list(centroids[:,0]))
        Y.extend(list(centroids[:,1]))
    spotcalls['CoordX'] = X
    spotcalls['CoordY'] = Y
    return spotcalls

def im2stage_wrapper(spotcalls):
    pos_spot_list = []
    for pos in spotcalls.posname.unique():
        pos_spot_list.append(spotcalls[spotcalls.posname==pos])
    with multiprocessing.Pool(30) as ppool:
        sys.stdout.flush()
        pfunc = partial(img2stage_coordinates)
        converted_spot_list = []
        for spots in ppool.imap(pfunc, pos_spot_list):
            converted_spot_list.append(spots)
            print(spots.posname.iloc[0],' Finished')
        ppool.close()
        sys.stdout.flush()
    spotcalls = pd.concat(converted_spot_list,ignore_index=True)
    return spotcalls

def progress_update(md,path=False):
    if path==True:
        from metadata import Metadata
        md = Metadata(md)
    finished = []
    non_finished = []
    import os
    for pos in os.listdir(os.path.join(md.base_pth,'codestacks')):
        try:
            processed = pickle.load(open(os.path.join(md.base_pth,'codestacks',pos,'processing.pkl'),'rb'))
        except:
            processed=[]
        if len(processed)==18:
            finished.append(pos)
        else:
            non_finished.append(pos)
    print(len(finished),' Positions Finished')
    print(len(non_finished),' Positions Not Finished')

def Display(image,title='Figure',cmap='inferno',figsize=(10,10),rel_min=0.1,rel_max=99.9,colorbar=False):
    img = image.copy()
    if rel_min>0:
        img_min = np.percentile(img.ravel(),rel_min)
        img[img<img_min]=img_min
    if rel_max<100:
        img_max = np.percentile(img.ravel(),99)
        img[img>img_max]=img_max
    plt.figure(figsize=figsize)
    plt.title(str(title))
    plt.imshow(img,cmap=cmap)
    if colorbar:
        plt.colorbar()
    plt.show()
    
def spot_dist(spotcalls,center=True):
    if isinstance(center,bool):
        center = [np.median(spotcalls.CoordX),np.median(spotcalls.CoordY)]
    print(center)
    dist = []
    X = spotcalls.CoordX
    Y = spotcalls.CoordY
    for i in range(len(spotcalls)):
        if i%100000==0:
            print(i)
        dist.append(sqrt( (center[0] - X.iloc[i])**2 + (center[1] - Y.iloc[i])**2 ))
    spotcalls['distance']=dist
    return spotcalls

def nuc_norm_list(spotcalls,md,center=True,nucstain=True):
    if isinstance(nucstain,bool):
        nucstain = [i for i in md.acqnames if 'nucstain' in i][0]
    if isinstance(center,bool):
        center = [np.median(spotcalls.CoordX),np.median(spotcalls.CoordY)]
    dist_list = []
    intensity_list = []
    for pos in spotcalls.posname.unique():
        print(pos)
        x,y = md.image_table[md.image_table.Position==pos].XY.iloc[0]
        im = md.stkread(Position=pos,Channel='DeepBlue',acq=nucstain)
        im = np.sum(im,axis=2)
        im_resize = resize(im,(int(2048*0.109),int(2048*0.109)),
                           mode='constant',anti_aliasing=True)
        for x_loc in range(im_resize.shape[1]):
                x_vector = im_resize[:,x_loc]
                for y_loc,pixel in enumerate(x_vector):
                    pix_x = x+x_loc
                    pix_y = y+y_loc
                    distance = sqrt( (center[0] - pix_x)**2 + (center[1] - pix_y)**2 )
                    dist_list.append(distance)
                    intensity_list.append(pixel)
    return(intensity_list,dist_list)

def nuc_hist(intensity_list,dist_list,nbins=100,rmin=0,rmax=1600):              
    bins = np.linspace(rmin,rmax,nbins)
    bins = bins.astype(int)
    hist = []
    intensity_array = np.array(intensity_list)
    for i in range(len(bins)-1):
        hist.append(np.sum(intensity_array[(dist_list>(bins[i]))&(dist_list<(bins[i+1]))]))
    return(hist,bins)

def generate_spatial_df(spotcalls,bins):
    genes = spotcalls.gene.unique()
    gene_dist = np.zeros((len(genes),len(bins)-1))
    for i,gene in enumerate(genes):
        temp_dist = spotcalls[spotcalls.gene==gene].distance
        gene_dist[i,:] = np.histogram(temp_dist,bins=bins)[0]
    step = (bins[1]-bins[0])/2
    disp_bins = bins+step
    disp_bins = disp_bins[0:99]
    gene_df = pd.DataFrame(gene_dist,index=genes,columns=disp_bins)
    return gene_df

def fix_channel_swap(md_path,hybes):
    import os
    for hybe in hybes:
        path = os.path.join(md_path,hybe)
        Channel = ['DeepBlue','FarRed','Orange']
        for pos in os.listdir(path):
            if 'Pos' in pos:
                for temp_file in os.listdir(os.path.join(path,pos)):
                    channel=[]
                    for temp_channel in Channel:
                        if temp_file.find(temp_channel)>0:
                            channel = temp_channel
                    if channel == 'DeepBlue':
                        new_file = temp_file.replace(channel,'FormallyDeepBlue')
                    elif channel == 'FarRed':
                        new_file = temp_file.replace(channel,'FormallyFarRed')
                    elif channel == 'Orange':
                        new_file = temp_file.replace(channel,'FormallyOrange')
                    else:
                        print(temp_file)
                        print(channel)
                        print(new_file)
                        raise Exception('channel didnt match?')
                    os.rename(os.path.join(path,pos,temp_file),os.path.join(path,pos,new_file))
        Channel = ['FormallyDeepBlue','FormallyFarRed','FormallyOrange']
        for pos in os.listdir(path):
            if 'Pos' in pos:
                for temp_file in os.listdir(os.path.join(path,pos)):
                    channel=[]
                    for temp_channel in Channel:
                        if temp_file.find(temp_channel)>0:
                            channel = temp_channel
                    if channel == 'FormallyDeepBlue':
                        new_file = temp_file.replace(channel,'Orange')
                    elif channel == 'FormallyFarRed':
                        new_file = temp_file.replace(channel,'DeepBlue')
                    elif channel == 'FormallyOrange':
                        new_file = temp_file.replace(channel,'FarRed')
                    else:
                        print(temp_file)
                        print(channel)
                        print(new_file)
                        raise Exception('channel didnt match?')
                    os.rename(os.path.join(path,pos,temp_file),os.path.join(path,pos,new_file))
def Display_mask(mask,figsize=[10,10],title='mask'):
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(colorize_segmented_image(mask))
    plt.show()
                    
def colorize_segmented_image(img, color_type='rgb'):
    """
    Returns a randomly colorized segmented image for display purposes.
    :param img: Should be a numpy array of dtype np.int and 2D shape segmented
    :param color_type: 'rg' for red green gradient, 'rb' = red blue, 'bg' = blue green
    :return: Randomly colorized, segmented image (shape=(n,m,3))
    """
    # get empty rgb_img as a skeleton to add colors
    rgb_img = np.zeros((img.shape[0], img.shape[1], 3))

    # make your colors
    num_cells = np.max(img)  # find the number of cells so you know how many colors to make
    colors = np.random.randint(0, 255, (num_cells, 3))
    if not 'r' in color_type:
        colors[:, 0] = 0  # remove red
    if not 'g' in color_type:
        colors[:, 1] = 0  # remove green
    if not 'b' in color_type:
        colors[:, 2] = 0  # remove blue

    regions = regionprops(img)
    for i in range(1, len(regions)):  # start at 1 because no need to replace background (0s already)
        rgb_img[tuple(regions[i].coords.T)] = colors[i]  # won't use the 1st color

    return rgb_img.astype(np.int)