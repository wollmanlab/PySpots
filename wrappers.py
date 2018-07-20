def multi_z_wrapper(posname, reg_ref='hybe1', zstart=5, k=2, zskip=4, zmax=26):
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
