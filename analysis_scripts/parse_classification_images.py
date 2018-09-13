from skimage.measure import regionprops, label
import pandas as pd
import numpy as np
from collections import defaultdict
def parse_classification_image(class_img, cstk, cvectors, genes, zindex):
    #class_imgs = data['class_img']
    #cstk = data['cstk']
    label2d = label((class_img+1).astype('uint16'), neighbors=8)
    properties = regionprops(label2d, (class_img+1).astype('uint16'))
    areas = []
    nclasses = []
    df_rows = []
    multiclass_sets = 0
    bit_values = defaultdict(list)
    for prop in properties:
        coords = prop.coords
        centroid = prop.centroid
        classes = list(set(prop.intensity_image.flatten())-set([0]))
        #classes = list(set(list())-set([0]))
        if len(classes)==0:
            print('Label with no classes.', end='')
            pdb.set_trace()
            continue
        elif not len(classes)==1:
            #print('Labels need to be broken apart more than one classification found per label.', end='')
            #print(classes)
            pdb.set_trace()
            multiclass_sets+=1
            continue
        else:
            #different_z.append(len(set(coords[:,2])))
            nclasses.append(len(classes))
            areas.append(prop.area)
        codeword_idx = classes[0]-1
        bits = np.where(cvectors[codeword_idx]==1)[0]
        
        spot_pixel_values = []
        for x, y in coords:
            cur_vals = cstk[x, y, bits]
            spot_pixel_values+=list(cur_vals)
            for idx, b in enumerate(bits):
                bit_values[b].append(cur_vals[idx])
            #norm_spot_pixel_values = spot_pixel_values
            #norm_spot_pixel_values = nstk[x, y, bits][0]
            #pdb.set_trace()
        df_rows.append([genes[codeword_idx], centroid, spot_pixel_values,
                        np.mean(spot_pixel_values), len(coords), codeword_idx, coords])
    print(multiclass_sets)
    df = pd.DataFrame(df_rows, columns=['gene', 'centroid', 'pixel_values', 'mean', 'npixels', 'cword_idx', 'coords'])
    return df, bit_values
def multi_z_class_parse_wrapper(f, cvectors, genes):
    data = np.load(f)
    cstks, nfs, class_imgs = data['cstks'].tolist(), data['norm_factors'].tolist(), data['class_imgs'].tolist()
    data.close()
    merged_df =[]
    for z, cstk in cstks.items():
        df, bvs = parse_classification_image(class_imgs[z], cstk, cvectors, genes, z)
        df['z'] = z
        merged_df.append(df)
    return pd.concat(merged_df, ignore_index=True)
