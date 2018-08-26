import numpy as np
import pickle
from metadata import Metadata
from skimage import io
from collections import Counter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("md_path", type=str, help="Path to root of imaging folder to initialize metadata.")
parser.add_argument("out_path", type=str, help="Path to save output.")
parser.add_argument("-n", "--nrepeats", type=int, dest="nrepeats", default=10, action='store', nargs=1, help="Number of time to repeat hot pixel finding.")
args = parser.parse_args()
print(args)
md = Metadata(args.md_path)
fnames_full = md.image_table[md.image_table.Channel.isin(['Orange', 'Green', 'FarRed'])].filename

candidates = []
nrepeats = args.nrepeats[0]
for i in range(nrepeats):
    print(i)
    fnames = np.random.choice(fnames_full, size=100, replace=False)
    stk = np.stack([io.imread(f) for f in fnames], axis=2)
    smean = np.mean(stk, axis=2).flatten()
    smin = np.min(stk, axis=2).flatten()
    sstd = np.std(stk, axis=2).flatten()
    #pixels = list(zip(smean, smin, sstd))
    hpixels = []
    for i, (minnie, meanie) in enumerate(zip(smin, smean)):
        if minnie > 5000 and meanie > 7500:
            hpixels.append(i)
    hpixels = np.array(hpixels)
    hpixels = np.unravel_index(hpixels, (2048, 2048))
    candidates.append(hpixels)
#     from sklearn.cluster import KMeans
#     kmeans = KMeans(n_clusters=4).fit(pixels)
#     label_counts = Counter(kmeans.labels_)
#     hpixel_cluster = label_counts.most_common(4)[3][0]
#     candidate_hot_pixels1 = 
#     candidates.append(candidate_hot_pixels1)

counts = Counter()
for c in candidates:
    c = list(zip(c[0], c[1]))
    counts.update(list(c))
hot_pixels = [k for k, v in counts.items() if v>nrepeats/4]
print(len(hot_pixels))
pickle.dump(hot_pixels, open(args.out_path, 'wb'))

