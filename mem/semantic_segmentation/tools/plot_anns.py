import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import shutil
from tqdm import tqdm
import cv2

PALETTE = np.array([[102, 179,  92],
       [ 14, 106,  71],
       [188,  20, 102],
       [121, 210, 214],
       [ 74, 202,  87],
       [116,  99, 103],
       [151, 130, 149],
       [ 52,   1,  87],
       [235, 157,  37],
       [129, 191, 187],
       [ 20, 160, 203]])  # (11, 3)

INDIR = "/storage/slurm/klenk/datasets/SS_final/anns/val/"
OUTDIR = "/home/wiss/klenk/Documents/GT_semseg/"
os.makedirs(OUTDIR, exist_ok=True)

for root, dirs, files in os.walk(INDIR, topdown=False):
    seq = root.split("/")[-1]

    outdir = os.path.join(OUTDIR, seq)
    os.makedirs(outdir, exist_ok=True)
    
    for f in files: 
        infile = os.path.join(root, f)
        annmap = cv2.imread(infile) # (H, W, 3)
        #  annmap_color = 

        for i in range(len(PALETTE)):
            mask = (annmap == i) # (H, W, 3)
            annmap = np.where(mask, PALETTE[i], annmap)

        annmap = annmap[:, :, ::-1]
        outfile = os.path.join(outdir, f)
        cv2.imwrite(outfile, annmap)
    print(root)