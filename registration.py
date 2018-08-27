import numpy
import numpy as np
from collections import defaultdict
from scipy.spatial import KDTree
from metadata import Metadata
from skimage import io
from skimage.feature import match_template, peak_local_max


def find_beads_3D(fnames_list, bead_template, match_threshold=0.75):
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
    hybe_names = ['hybe1', 'hybe2', 'hybe3', 'hybe4', 'hybe5', 'hybe6']
    #fnames_dict = {k.split('_')[0]:v for k, v in fnames_dict.items()}
#     fish_md = Metadata(md_path, md_name='Metadata.dsv')
#     pos = fish_md.posnames[1]
#     fish_md.image_table = fish_md.image_table[fish_md.image_table.Position==pos]
    #tforms = {reg_reference: {'tvec': numpy.array([0, 0])}}
    #ref_fnames = fnames_dict.pop(reg_reference)
    ref_stk = numpy.stack([io.imread(i) for i in fnames_list], axis=2).astype('float')
    ref_match = match_template(ref_stk, bead_template)
    ref_beads = peak_local_max(ref_match, threshold_abs=match_threshold)
    ref_substks, keep_list = fetch_substacks(ref_stk, ref_beads, w=3)
    ref_beads = ref_beads[keep_list, :]
    #del ref_stk
    return ref_beads, ref_substks

def processStk(stk, npeaks=200):
    stk = np.subtract(stk, np.min(stk, axis=2)[:,:,np.newaxis])
    stk = gaussian_filter(stk, (0.9, 0.9, 0.9))
    peaks = peak_local_max(stk, num_peaks=npeaks)
    return stk, peaks

def fetch_substacks(stk, peaks, w=3, sz = 7):
    imsz = stk.shape
    substks = {}
    keep_list = []
    for ii, p in enumerate(peaks):
        if (p[0] <= w) or (p[0] >= imsz[0]-w):
            #pop_list.append(ii)
            continue
        if (p[1] <= w) or (p[1] >= imsz[1]-w):
            #pop_list.append(ii)
            continue
        if (p[2] <= w) or (p[2] >= imsz[2]-w):
            #pop_list.append(ii)
            continue
        substk = stk[p[0]:p[0]+sz, p[1]:p[1]+sz, p[2]:p[2]+sz]
        substks[tuple(p)] = substk
        keep_list.append(ii)
    #pop_list = {tuple(k):0 for k in pop_list}
    return substks, keep_list

def find_average_bead_profile(stk, w, sz):
    fstk, peaks = processStk(stk.copy())
    s = fetch_substacks(stk, peaks)
    beads = []
    for p in peaks:
        p = tuple(p)
        if p[2] == 18:
            beads.append(s[p])
    ave_bead = np.mean(beads, 0)
    ave_bead = ave_bead/np.amax(ave_bead)

def reject_outliers(data, m=3):
    d = np.abs(data - np.median(data, axis=0))
    mdev = np.median(d, axis=0)
    s = [True if sum(i)<m else False for i in d]
    data = [d for idx, d in enumerate(data) if s[idx]]
    return data

def find_tforms(beads, reg_ref_hybe = 'hybe1', max_dist = 30, verbose=False):
    """
    Process putative bead coordinates to find registration shifts between hybes.
    
    Parameters
    ----------
    beads : dict
        Dictionary of bead coordinates for each hybe
    reg_ref_hybe : str - default - 'hybe1'
    max_dist : int
        Maximum distance for two beads to be linked between hybes
    
    Returns
    -------
    Not well defined yet
    
    Notes - not very clean function - future work cleans it up and probably 
    implements a better way of finding Z fits.
    
    """
    if 'nucstain' in beads:
        beads.pop('nucstain')
    hybe_names = []
    tforms = {reg_ref_hybe: {'tvec': numpy.array([0, 0])}}
    ref_beads = beads.pop(reg_ref_hybe)
    bead_sets = defaultdict(dict)
    # Find beads in nonreference hybes
    tree = KDTree(ref_beads[:, 0:2]) #Data structure for finding close beads quickly
    for h, current_beads in beads.items():
        hybe_names.append(h)
        if verbose:
            print('Registering:', h)
        # Reject bead candidates that are >max_dist from any bead in reference
        dists, idxes = tree.query(current_beads[:, 0:2])
        shifts = []
        # Store all bead candidates in nonreference with their closest
        # cognate bead in the reference
        for b_i, b in enumerate(current_beads):
            if dists[b_i]<max_dist:
                bead_sets[tuple(ref_beads[idxes[b_i]])][h] = b
    # Only keep beads found in all hybes 
    master_beads = {}
    for k, v in bead_sets.items():
        if len(v)==len(hybe_names):
            master_beads[k] = v  
    tforms_xy = defaultdict(list)
    tforms_z = defaultdict(list)
    tforms_z_raw = defaultdict(list)
    # Iterate over all beads found in all hybes and fit gaussians then
    # find the shift between the reference and nonreference
    for k, hybe_dict in master_beads.items():
        #if str(k) in ref_substks:
            #sstk = ref_substks[str(k)]
            #ref_fit = gaussfitter.gaussfit(sstk[:,:,3])
        ref_fit = (k[0], k[1])
            #print(ref_fit)
        for h, dest_bead in hybe_dict.items():
                #if str(tuple(dest_bead)) in destination_substks[h]:
            try:
                    #sstk = destination_substks[h][str(tuple(dest_bead))]
                    #dest_fit = gaussfitter.gaussfit(sstk[:,:,3])
                    #dest_fit = (dest_bead[0]+dest_fit[2]-3, dest_bead[1]+dest_fit[3]-3)
                dest_fit = (dest_bead[0], dest_bead[1])
                tforms_xy[h].append(np.subtract(ref_fit, dest_fit))
                tforms_z[h].append(np.subtract(k[2], dest_bead[2]))
                tforms_z_raw[h].append((k[2], dest_bead[2]))
            except:
                continue


    # Attempt to find robust mean of transformation to map beads between
    # source/destination if fails use regular mean
    tforms_z_Ave = defaultdict(list)
    try:
        tforms_z_r = {h: huber(v).astype('int') for h, v in tforms_z.items()}
    except:
        tforms_z_r = {h: np.round(np.mean(v)).astype('int') for h, v in tforms_z.items()}
    #tforms_z_Ave = {h: np.mean(reject_outliers(v)) for h, v in tforms_z.items()}
    tforms_xy = {h: np.nanmean(reject_outliers(v), axis=0) for h, v in tforms_xy.items()}
    error_xy = 0
    error_z = 0
    nbeads = len(master_beads)
    return nbeads, tforms_xy, tforms_z_r, error_xy, error_z, master_beads

from codestack_creation import tform_image
def pseudo_bead_stacks_byposition(posname, md_pth, tforms_xy, tforms_z, zstart=6, k=2, reg_ref='hybe1'):
    hybe_names = ['hybe1', 'hybe2', 'hybe3', 'hybe4', 'hybe5', 'hybe6', 'hybe7', 'hybe8', 'hybe9']
    md = Metadata(md_pth)
    xy = tforms_xy
    z = tforms_z
    z = {k: int(np.round(np.mean(v))) for k, v in z.items()}
    z[reg_ref] = 0
    xy[reg_ref] = (0,0)
    cstk = []
    chan='DeepBlue'
    for hybe in hybe_names:
        t = xy[hybe]
        zindexes = list(range(zstart-z[hybe]-k, zstart-z[hybe]+k+1))
        print(zindexes)
        zstk = md.stkread(Channel=chan, hybe=hybe, Position=posname, Zindex=zindexes)
        zstk = zstk.max(axis=2)
        zstk = tform_image(zstk, chan, t)
        cstk.append(zstk)
        del zstk
    cstk = np.stack(cstk, axis=2)
    #nf = np.percentile(cstk, 90, axis=(0, 1))
    return cstk

# Methods to use cross correlation (example multiprocessign call below)
#results = ppool.starmap(wrappadappa_bead_xcorr, zip(posnames, itertools.repeat(md_pth, len(posnames))))
#seed_tforms = {p:k for p,k in results}
import imreg_dft as ird
def hybe_composite(md_pth, posname, channels = ['DeepBlue'],
                   zindexes=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14], nhybes = 9):
    """
    Create maximum projection composites from each hybe.
    """
    hybe_names = ['hybe'+str(i) for i in range(1, nhybes+1)]
    md = Metadata(md_pth)
    if zindexes is None:
        zindexes = list(md.image_table.Zindex.unique())
    hybe_composites = {}
    for h in hybe_names:
        hybe_stk = md.stkread(Channel=channels, Position=posname, Zindex=zindexes, hybe=h)
        hybe_stk = hybe_stk.max(axis=2)
        hybe_composites[h] = hybe_stk
    return hybe_composites

def xcorr_hybes(hybe_dict, reg_ref = 'hybe1'):
    """
    Find the translation xcorr between hybes.
    """
    tvecs = {}
    for h, img in hybe_dict.items():
        if h == reg_ref:
            tvecs[h] = (0,0)
        else:
            xcorr_result = ird.translation(hybe_dict[reg_ref], img)
            tvecs[h] = xcorr_result['tvec']
    return tvecs


def wrappadappa_bead_xcorr(posname, md_pth):
    bead_projections = hybe_composite(md_pth, posname, channels=['DeepBlue'], zindexes=None, nhybes=9)
    tvecs = xcorr_hybes(bead_projections)
    return posname, tvecs


# Functions to use new bead fitting routine (use example)
# func_inputs = [(bead_pos_dicts[pos], seed_tforms[pos]) for pos in posnames]
# bead_pos_dicts is dictionary of dictionarys
    #pos:hybe_dict where hybe_dict is hybe_name:bead_array
# seed_tforms is dictionary created by above functions

# ppool = multiprocessing.Pool(24)
# results = ppool.starmap(optimize_tforms, func_inputs)
# good_positions = {}
# r = []
# n = []
# q = []
# for pos, (tvec, quals) in zip(posnames, results):
#     residuals = [i['residual'] for i in quals.values()]
#     nbeads = [i['nbeads'] for i in quals.values()]
#     ratio = [i['bead_outlier_ratio'] for i in quals.values()]
#     r += residuals
#     n += nbeads
#     q += ratio
#     if np.amax(residuals) > 1.5:
#         continue
#     if np.amin(ratio) < 0.8:
#         continue
#     good_positions[pos] = (tvec, quals)
# gposnames = list(good_positions.keys())

# print('Number good positoins: ', len(good_positions))


from scipy.spatial import KDTree
import scipy
def find_pair_error(tvect, beads1, beads2):
    #global naccepted, dists, naccepted2, residual
    beads2_reg = beads2+tvect
    tree = KDTree(beads1)
    b2_pair = [tree.query(p) for p in beads2_reg]
    dists, idx = zip(*b2_pair)
    b2_pair = [beads1[i] for i in idx]
    #list(zip(dists, beads2, b2_pair))
    naccepted, idx = reject_outliers(dists)
#     if len(naccepted)/len(dists) < 0.8:
#         print('Warning lots of beads rejected')
    naccepted2, idx = reject_outliers(naccepted)
#     if len(naccepted2)/len(dists) < 0.8:
#         print('Warning lots of beads rejected')
    residual = np.abs(np.mean(naccepted2))
    return residual

def find_pair_error2(tvect, beads1, beads2):
    #global naccepted, dists, naccepted2, residual
    beads2_reg = beads2+tvect
    tree = KDTree(beads1)
    b2_pair = [tree.query(p) for p in beads2_reg]
    dists, idx = zip(*b2_pair)
    b2_pair = [beads1[i] for i in idx]
    #list(zip(dists, beads2, b2_pair))
    naccepted, idx = reject_outliers(dists)
#     if len(naccepted)/len(dists) < 0.8:
#         print('Warning lots of beads rejected')
    naccepted2, idx = reject_outliers(naccepted)
#     if len(naccepted2)/len(dists) < 0.8:
#         print('Warning lots of beads rejected')
    residual = np.abs(np.mean(naccepted2))
    return residual, len(naccepted2), len(dists)


def reject_outliers(data, m=2):
    """
    Reject outliers for robust distance estimation.
    (Bead candidates could be incorrect or their cognate pair could be
    missing in the cognate hybe)
    """
    data = np.array(data)
    good_idx = abs(data - np.mean(data)) < m * np.std(data)
    return data[good_idx], good_idx
    
def optimize_tforms(bead_dict, seed_tforms, reg_ref='hybe1', verbose=False):
    """
    New registration method. Find seed tforms by xcorrelation then optimize with 
    the bead candidate coordinates.
    """
    #pos = posnames[11]
    nucs = bead_dict.pop('nucstain')
    reg_ref = 'hybe1'
    opt_tforms = {}
    tform_quality_metrics = defaultdict(dict)
    bead_ref = bead_dict.pop(reg_ref)
    for h, bead_dest in bead_dict.items():
        initial = seed_tforms[h]
        initial = (initial[0], initial[1], 0)
        tvect = scipy.optimize.fmin(find_pair_error, initial,
                                 args=(bead_ref, bead_dest), disp=verbose)
        opt_tforms[h] = tvect
        residual, naccepted2, dists = find_pair_error2(tvect, bead_ref, bead_dest)
        tform_quality_metrics[h]['nbeads'] = naccepted2
        tform_quality_metrics[h]['bead_outlier_ratio'] = naccepted2/dists
        tform_quality_metrics[h]['residual'] = residual
    opt_tforms[reg_ref] = np.array((0,0,0))
    bead_dict[reg_ref] = bead_ref
    bead_dict['nucstain'] = nucs
    return opt_tforms, tform_quality_metrics
    