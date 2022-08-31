import pytest
import scipy
from loguru import logger
import datetime
import numpy as np
from devel.patches3d import make_patch, place_patch_on_tube, noises, random_rotate_volume, random_direction_vector, insert_patch_into_volume
from devel.patches_dataset import make_dataset
from matplotlib import pyplot as plt
import skimage.transform
import imma

import io3d
import io3d.datasets
from pathlib import Path
import copy


def test_patches():

    make_dataset(Path("."), "train", 3, 2)
    make_dataset(Path("."), "test",dataset_image_number=3, repeat_number=5)
    make_dataset(Path("."), "val", dataset_image_number=3, repeat_number=5)
    # io3d.datasets.joinp()
    # # ct_fn = "/storage/plzen4-ntis/projects/korpusy_cv/mjirik_pilsen_pigs/Tx018D_Ven.mhd"
    # datai = 1
    # ct_fn = io3d.joinp(f"biomedical/orig/pilsen_pigs/test/PP_{datai:04d}/PATIENT_DICOM/PP_{datai:04d}.mhd")


    # %% md

    # %%

    # %%
    ## No idea why to do that
    # pts = np.asarray(np.nonzero(patch3d > background_density_hu))
    # mx = np.max(pts, axis=1)
    # mn = np.min(pts, axis=1)
    # margin = int(np.mean(mx) * 0.05)
    # margin
    # # %%
    # patch3d.shape
    # # %%
    # mx
    # # %%
    # patch3dm = (np.ones(mx + 2 * margin) * background_density_hu).astype(np.int16)
    # # %%
    # patch3dm[
    #     margin:mx[0] + margin, margin:mx[1] + margin, margin:mx[2] + margin
    # ] = patch3d[
    #     mn[0]:mx[0], mn[1]:mx[1], mn[2]: mx[2]
    #     ]
    # # %%
    # plt.imshow(np.max(patch3dm, axis=0))
    # %%



