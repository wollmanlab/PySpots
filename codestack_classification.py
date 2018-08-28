# Robert Foreman - Wollman Lab - 2018

def classify_codestack(cstk, norm_vector, codeword_vectors, csphere_radius=0.5176):
    """
    Pixel based classification of codestack into gene_id pixels.
    
    Parameters
    ----------
    cstk : ndarray
        Codestack (y,x,codebit)
    norm_vector : array
        Array (ncodebits,1) used to normalization intensity variation between codebits.
    codeword_vectors : ndarray
        Vector Normalized Array (ncodewords, ncodebits) specifying one bits for each gene
    csphere_radius : float
        Radius of ndim-sphere serving as classification bubble around codeword vectors
    
    Returns
    -------
    class_img : ndarray
        Array (y,x) -1 if not a gene else mapped to index of codeword_vectors
    """
    cstk = cstk.copy()
    cstk = cstk.astype('float32')
    # Normalize for intensity difference between codebits
    cstk = np.divide(cstk, norm_vector)
    # Prevent possible underflow/divide_zero errors
    np.place(cstk, cstk<=0, 0.01)
    # Fill class img one column at a time
    class_img = defaultdictnp.empty((cstk.shape[0], cstk.shape[1]))
    for i in range(cstk.shape[0]):
        v = cstk[i, :, :]
        # l2 norm note codeword_vectors should be prenormalized
        v = normalize(v, norm='l2')
        # Distance from unit vector of codewords and candidate pixel codebits
        d = distance_matrix(codeword_vectors, v)
        # Check if distance to closest unit codevector is less than csphere thresh
        dmin = np.argmin(d, axis=0)
        dv = [i if d[i, idx]<csphere_radius else -1 for idx, i in enumerate(dmin)]
        class_img[i, :] = dv
    return class_img#.astype('int16')

def mean_one_bits_file(cstk, class_img):#, nbits = 18):
    """
    Calculate average intensity of classified pixels per codebits.
    
    Parameters
    ----------
    cstk : ndarray
        Codestack
    class_img : ndarray
        classification image
    nbits : int
        Number of codebits (maybe just make it equal to cstk shape?)
    
    Returns
    -------
    mean_bits : array
        Mean of classified pixels for each codebit
    """
    bitvalues = defaultdict(list)
    cstk = cstk.astype('float32')
    nbits = cstk.shape[2]
    for i in range(cvectors.shape[0]):
        x, y = np.where(class_img==i)
        if len(x) == 0:
            continue
        onebits = np.where(cvectors[i,:]==1)[0]
        if len(onebits)<1:
            continue
        for i in onebits:
            bitvalues[i].append(np.mean(cstk[x, y, i]))
    mean_bits = np.array([robust_mean(bitvalues[i]) for i in range(nbits)])
    return mean_bits

def robust_mean(x):
    return np.average(x, weights=np.ones_like(x) / len(x))

def reclassify_file(f, nfactor):
    """
    Wrapper for iterative classification.
    """
    pth, pos = os.path.split(f)
    print(pos)
    a = pickle.load(open(os.path.join(base_path, p), 'rb'))
    cstk = a['cstk']
    #class_img = a['class_img']
    new_class_img = classify_codestack(cstk, nfactor, nvectors)  
    pickle.dump({'cstk': cstk, 'nf': nfactor,
                 'class_img': new_class_img}, open(f, 'wb'))
    return mean_one_bits_file(cstk, new_class_img)
#     np.savez(os.path.join(base_path, p), cstk=cstk,
#              class_img=new_class_img, norm_factor=nfactor)
                    
def classify_file(f, nfactor):
    """
    Wrapper for classify_codestack. Can change this instead of function if 
    intermediate file storage ever changes.
    """
    try:
        pth, pos = os.path.split(f)
        #print(pos)
        data = np.load(f)
        try:
            cstks, nfs, class_imgs = data['cstks'].tolist(), data['norm_factors'].tolist(), data['class_imgs'].tolist()
        except:
            cstks, nfs = data['cstks'].tolist(), data['norm_factors'].tolist()
            class_imgs = {}
        for z in nfs.keys():
            cstk = cstks[z]
            new_class_img = classify_codestack(cstk, nfactor, nvectors)
            class_imgs[z] = new_class_img
            new_nf = mean_one_bits_file(cstk, new_class_img)
            nfs[z] = new_nf
            
        np.savez(os.path.join(pth, pos), cstks=cstks, norm_factors=nfs, class_imgs=class_imgs)
    except:
        return f
def mean_nfs_npz(fname):
    """
    Iterate through codestacks and average norm factors.
    """
    data = np.load(fname)
    nfs = data['norm_factors'].tolist()
    return np.nanmean([n for k, n in nfs.items()], axis=0)

def load_codestack_from_npz(fname):
    """
    Load saved codestack data.
    
    Parameters
    ----------
    fname : str, filestream
    
    Returns
    -------
    cstk : dict
        Dictionary of (y,x,nbits) arrays at different Z's
    nfs : array
        
    class_imgs : dict
        Dictionary of (y,x) arrays at different Z's of gene classifications
    """
    data = np.load(fname)
    cstk = data['cstks'][()]
    class_imgs = data['class_imgs'][()]
    nfs = data['norm_factors'][()]
    return cstk, nfs, class_imgs

