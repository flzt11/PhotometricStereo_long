from __future__ import print_function

import numpy as np
import time
import pris


DATA_FOLDERNAME = './cv/data/bunny/bunny_lambert/'    # Lambertian diffuse with cast shadow
#DATA_FOLDERNAME = './data/bunny/bunny_lambert_noshadow/'    # Lambertian diffuse without cast shadow

LIGHT_FILENAME = './cv/data/bunny/lights.npy'
MASK_FILENAME = './cv/data/bunny/mask.png'
GT_NORMAL_FILENAME = './cv/data/bunny/gt_normal.npy'


# Photometric Stereo
rps = pris.cv.ps.PS()
pris.utils.load_all(rps, mask_filename=MASK_FILENAME, light_filename=LIGHT_FILENAME, data_foldername=DATA_FOLDERNAME)
pris.utils.solve_and_save(rps, normal_map_filename="./est_normal")
pris.utils.psutil.evaluate_and_display(rps, GT_NORMAL_FILENAME)
