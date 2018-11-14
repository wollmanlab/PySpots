import os
import numpy
import pickle
from skimage import io
import multiprocessing
from itertools import repeat
import multiprocessing as mp
from functools import partial
from metadata import Metadata
from scipy.spatial import KDTree
from collections import defaultdict
from skimage.feature import match_template, peak_local_max
from skimage.filters import gaussian
from skimage.transform import resize

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("md_path", type=str, help="Path to where your images are unless they are in Analysis/deconvolved/.",default = 'None')
    parser.add_argument("analysis_path", type=str, help="Path to where you want your data saved.")
    parser.add_argument("-p", "--nthreads", type=int, dest="ncpu", default=6, action='store', help="Number of cores to utilize (default 6).")
    parser.add_argument("-z", "--zindexes", type=int, nargs='*', action='store', dest="zindexes", default=-1)
    parser.add_argument("-b", "--beadprofile", type=str, action='store', dest="ave_bead_path",
                        default='/bigstore/Images2018/Zach/FISH_Troubleshooting/Transverse_Second_2018Aug24/Analysis/results/Avg_Bead.pkl')
    args = parser.parse_args()
    print(args)

def keep_which(stk, peaks, w=3, sz = 7):
    """
    Only necessary if grabbing substks to prevent accessing out of bounds pixels.
    """
    imsz = stk.shape
    keep_list = []
    for ii, p in enumerate(peaks):
        if (p[0] <= w) or (p[0] >= imsz[0]-w):
            continue
        if (p[1] <= w) or (p[1] >= imsz[1]-w):
            continue
        if (p[2] <= w) or (p[2] >= imsz[2]-w):
            continue
        keep_list.append(ii)
    return keep_list

def find_beads_3D(posname, md_table, bead_template, match_threshold=0.65, upsamp_factor = 5):
    """
    3D registration from sparse bead images.
    Parameters
    ----------
    fnames_dict : dict
        Dictionary (hybe name:list filenames) to load bead images
    bead_template : numpy.array
        Array of normalized values of 'average bead' intensities.
        This helps to not pick up noisy spots/hot pixels.
    ref_reference : str - default('hybe1')
        Which hybe is the reference destination to map source hybes onto.
    max_dist : int - default(50)
        The maximum distanced allowed between beads found when pairing 
        beads between multiple hybes.
    match_threshold : float - default(0.75)
        The similarity threshold between bead_template and images to 
        consider potential beads.
    Returns
    -------
    tforms : dict
        Dictionary of translation vectors (x, y, z)? maybe (y, x, z)
    """
    #hybe_names = ['hybe1', 'hybe2', 'hybe3', 'hybe4', 'hybe5', 'hybe6']
    ref_stk = numpy.stack([io.imread(i) for i in fnames_list], axis=2).astype('float')
    ref_match = match_template(ref_stk, bead_template, pad_input=True)
    ref_beads = peak_local_max(ref_match, threshold_abs=match_threshold)
    upsamp_bead = resize(bead_template[2:5, 2:5, 1:4],
                         (3*upsamp_factor, 3*upsamp_factor, 3*upsamp_factor))
    subpixel_beads = []
    for y, x, z in ref_beads:
        substk = ref_stk[y-5:y+6, x-5:x+6, z-2:z+3]
        if substk.shape[0] != 11 or substk.shape[1] != 11:
            continue # candidate too close to edge
        try:
            upsamp_substk = resize(substk,
                                   (substk.shape[0]*upsamp_factor,
                                    substk.shape[1]*upsamp_factor,
                                    substk.shape[2]*upsamp_factor))
        except:
            continue
        bead_match = match_template(upsamp_substk,
                                    upsamp_bead, pad_input=True)
        yu, xu, zu = numpy.where(bead_match==bead_match.max())
        yu = (yu[0]-int(upsamp_substk.shape[0]/2))/upsamp_factor
        xu = (xu[0]-int(upsamp_substk.shape[1]/2))/upsamp_factor
        zu = (zu[0]-int(upsamp_substk.shape[2]/2))/upsamp_factor
        ys, xs, zs = (yu+y, xu+x, zu+z)
        subpixel_beads.append((ys, xs, zs))
#         if len(subpixel_beads)<1:
#             subpixel_beads = numpy.array([])
#         else:
#             subpixel_beads = numpy.stack(subpixel_beads, axis=0)
    return subpixel_beads
#     if len(subpixel_beads)==0:
#         subpixel_beads = numpy.array([])
#     else:
#         subpixel_beads = numpy.stack(subpixel_beads, axis=0)
#     return subpixel_beads
#     return ref_beads

def add_bead_data(bead_dicts, ave_bead, Input):
    image_table = Input['image_table']
    pos = Input['posname']
    print('starting', pos)
    if pos in bead_dicts:
        bead_dict = bead_dicts[pos]
    else:
        bead_dict = {}
    for h in fnames_dict.keys():
        #convert to hybe1 not hybe1_4
        H = h.split('_')[0]
        if H in bead_dict:
            continue
        else:
            beads = find_beads_3D(fnames_dict[h], ave_bead)
            bead_dict[H] = beads
    print('Finished', pos)
    return bead_dict, pos
    #pickle.dump(bead_dict, open(os.path.join(bead_path,pos), 'wb'))
    
if __name__ == '__main__':
    #Setting up paths
    analysis_path = args.analysis_path
    md_path = args.md_path
    zindexes = args.zindexes
    ncpu = args.ncpu
    ave_bead_path = args.ave_bead_path
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)
    if md_path == 'None':
        deconvolved_path = os.path.join(analysis_path,'deconvolved')
        if not os.path.exists(deconvolved_path):
            print('analysis path doesnt have your metadata')
            print('add path to metadata after analysis path')
    else:
        deconvolved_path = md_path
#     bead_path = os.path.join(analysis_path,'beads')
#     if not os.path.exists(bead_path):
#         os.makedirs(bead_path)
    results_path = os.path.join(analysis_path,'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    #Retreiving file names and position names
    md = Metadata(deconvolved_path)
    md.image_table = md.image_table[[True if 'hybe' in i else False for i in md.image_table.acq]]
    posnames = md.posnames
    print('posnames loaded')
    hybe_list = sorted([i.split('_')[0] for i in md.acqnames if 'hybe' in i])
    if zindexes == -1:
        Input = [ {'image_table': md.image_table, 'posname': pos} for pos in posnames]
    else:
        fnames_dicts = [md.stkread(Channel='DeepBlue', Position=pos,
                           fnames_only=True, groupby='acq', 
                          hybe=hybe_list, Zindex=zindexes) for pos in posnames]
    print('fnames loaded')
#     Input = list()
#     for i in range(len(fnames_dicts)):
#         dictionary = defaultdict(dict)
#         dictionary['fname_dicts'] = fnames_dicts[i]
#         dictionary['posnames']= posnames[i]
#         Input.append(dictionary)
        
    Ave_Bead = pickle.load(open(ave_bead_path, 'rb'))
    Ave_Bead = Ave_Bead[:,:, 3:] # Only use the top half so that i can match things near coverslip better
    bead = numpy.zeros((7, 7, 5))
    bead[3, 3, 2] = 1
    bead = gaussian(bead, (1.5, 1.5, 0.85))
    Ave_Bead = bead/bead.max()
    #Setting up parrallel pool
    os.environ['MKL_NUM_THREADS'] = '3'
    os.environ['GOTO_NUM_THREADS'] = '3'
    os.environ['OMP_NUM_THREADS'] = '3'
    #Finding Beads
    if os.path.exists(os.path.join(results_path,'beads.pkl')):
        bead_dicts = pickle.load(open(os.path.join(results_path,'beads.pkl'), 'rb'))
    else:
        bead_dicts = defaultdict(dict)
    pfunc = partial(add_bead_data,bead_dicts,Ave_Bead)
    with multiprocessing.Pool(ncpu) as p:
        for Bead_dict,Pos in p.imap(pfunc, Input, chunksize=1):
            bead_dicts[Pos] = Bead_dict
            pickle.dump(bead_dicts, open(os.path.join(results_path,'beads.pkl'), 'wb'))
        
#    with mp.Pool(ncpu) as ppool:
#        ppool.starmap(add_bead_data, zip(fnames_dicts, posnames, repeat(Ave_Bead, len(posnames))))
   #Combining beads to one file
#    bead_dicts = defaultdict(dict)
#    for files in os.listdir(bead_path):
#        fi = os.path.join(bead_path,files)
#        d = pickle.load(open(fi, 'rb'))
#        d = {k.split('_')[0]:v for k, v in d.items()}
#        bpth, pos = os.path.split(fi)
#        bead_dicts[pos].update(d)
#    pickle.dump(bead_dicts, open(os.path.join(results_path,'beads.pkl'), 'wb'))
