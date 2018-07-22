import numpy
import numpy as np
def multi_z_pseudo_maxprjZ_wrapper(posname, reg_ref='hybe1', zstart=5, k=2, zskip=4, zmax=26):
    codestacks = {}
    norm_factors = {}
    class_imgs = {}
    for z_i in list(range(zstart, zmax, zskip)):
        cstk, nf = pseudo_maxproject_positions_and_tform(posname, tforms_xy, tforms_z, zstart=z_i)
        codestacks[z_i] = cstk.astype('uint16')
        norm_factors[z_i] = nf
        class_imgs = np.empty((cstk.shape[0], cstk.shape[1]))
    np.savez(os.path.join(cstk_save_dir, posname), cstks=codestacks, 
            norm_factors = norm_factors, class_imgs = class_imgs)


import shlex
import os
import pickle
from subprocess import check_output
from collections import defaultdict
from registration import find_tforms

def load_beads_find_tforms(bead_pth):
    bead_filez = check_output(shlex.split('find {0} -type f'.format(bead_pth))).decode().split('\n')
    beads_dict = defaultdict(dict)
    for f in bead_filez:
        if len(f)==0:
            continue
        pth, h = os.path.split(f)
        h = h.split('_')[0]
        f = pickle.load(open(f, 'rb'))
        for pos, beads in f:
            beads_dict[pos][h] = beads
    posnames = beads_dict.keys()
    tforms_xy = {}
    tforms_z = {}
    mbs = {}
    for pos in posnames:
        beads = beads_dict[pos]
        nbeads, txy, tz, exy, ezy, master_beads = find_tforms(beads, verbose=False)
        mbs[pos] = master_beads
        tforms_xy[pos] = txy
        tforms_z[pos] = tz
    return tforms_xy, tforms_z, mbs
