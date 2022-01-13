from tqdm import tqdm
import cv2
import random
from skimage.filters import threshold_otsu
def find_hot_pixels(md,n_pos=5,std_thresh=3,n_acqs=5,kernel_size=3):
    if kernel_size%2==0:
        kernel_size = kernel_size+1
    kernel = np.ones((kernel_size,kernel_size))
    kernel[int(kernel_size/2),int(kernel_size/2)] = 0
    kernel = kernel/np.sum(kernel)
    X = []
    Y = []
    acqs = [i for i in md.image_table.acq.unique() if 'hybe' in i]
    poses = md.image_table[md.image_table.acq.isin(acqs)].Position.unique()
    hot_pixel_dict = {}
    if len(poses)>n_pos:
        poses = random.sample(list(poses),n_pos)
    for pos in tqdm(poses):
        pos_md =  md.image_table[md.image_table.Position==pos]
        acqs = pos_md.acq.unique()
        if len(acqs)>n_acqs:
            acqs = random.sample(list(acqs),n_acqs)
        for acq in acqs:
            hot_pixel_dict[acq] = {}
            channels = pos_md[pos_md.acq==acq].Channel.unique()
            channels = set(list(channels)).intersection(['FarRed','Orange'])
            for channel in channels:
                img = np.average(md.stkread(Position=pos,Channel=channel,acq=acq),axis=2)
                bkg_sub = img-cv2.filter2D(img,-1,kernel)
                avg = np.average(bkg_sub)
                std = np.std(bkg_sub)
                thresh = (avg+(std_thresh*std))
                loc = np.where(bkg_sub>thresh)
                X.extend(loc[0])
                Y.extend(loc[1])
    img = np.histogram2d(X,Y,bins=2048,range=[[0,2048],[0,2048]])[0]
    loc = np.where(img>threshold_otsu(img))
    return img,loc,X,Y