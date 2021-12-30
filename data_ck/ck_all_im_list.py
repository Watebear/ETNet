import os
import glob

im_root = '/home/dddzz/workspace/Datasets/CK+/cohn-kanade-images'
outfile = open('all_im_list.txt', 'w')

subs = glob.glob(im_root+'/*')
for sub in subs:
    seqs = glob.glob(sub+'/*')
    for seq in seqs:
        imgs = glob.glob(seq+'/*.png')
        for img in imgs:
            img = img.replace(im_root+"/", "")
            outfile.write(img+'\n')

outfile.close()