#!/usr/bin/env python
# -*- coding: utf-8 -*-


from ..utils import psutil
import numpy as np
from sklearn.preprocessing import normalize

DATA_FOLDERNAME = './cv/data/bunny/bunny_lambert/'  # Lambertian diffuse with cast shadow
LIGHT_FILENAME = './cv/data/bunny/lights.npy'
MASK_FILENAME = './cv/data/bunny/mask.png'
GT_NORMAL_FILENAME = './cv/data/bunny/gt_normal.npy'

class PS(object):


    def __init__(self):
        self.M = None   # measurement matrix in numpy array
        self.L = None   # light matrix in numpy array
        self.N = None   # surface normal matrix in numpy array
        self.height = None  # image height
        self.width = None   # image width
        self.foreground_ind = None    # mask (indices of active pixel locations (rows of M))
        self.background_ind = None    # mask (indices of inactive pixel locations (rows of M))

    def load_lighttxt(self, filename=None):
        """
        Load light file specified by filename.
        The format of lights.txt should be
            light1_x light1_y light1_z
            light2_x light2_y light2_z
            ...
            lightf_x lightf_y lightf_z

        :param filename: filename of lights.txt
        """
        self.L = psutil.load_lighttxt(filename)

    def load_lightnpy(self, filename=None):
        """
        Load light numpy array file specified by filename.
        The format of lights.npy should be
            light1_x light1_y light1_z
            light2_x light2_y light2_z
            ...
            lightf_x lightf_y lightf_z

        :param filename: filename of lights.npy
        """
        self.L = psutil.load_lightnpy(filename)

    def load_images(self, foldername=None, ext=None):
        """
        Load images in the folder specified by the "foldername" that have extension "ext"
        :param foldername: foldername
        :param ext: file extension
        """
        self.M, self.height, self.width = psutil.load_images(foldername, ext)

    def load_npyimages(self, foldername=None):
        """
        Load images in the folder specified by the "foldername" in the numpy format
        :param foldername: foldername
        """
        self.M, self.height, self.width = psutil.load_npyimages(foldername)

    def load_mask(self, filename=None):
        """
        Load mask image and set the mask indices
        In the mask image, pixels with zero intensity will be ignored.
        :param filename: filename of the mask image
        :return: None
        """
        if filename is None:
            raise ValueError("filename is None")
        mask = psutil.load_image(filename=filename)
        mask = mask.reshape((-1, 1))
        self.foreground_ind = np.where(mask != 0)[0]
        self.background_ind = np.where(mask == 0)[0]

    def disp_normalmap(self, delay=0):
        """
        Visualize normal map
        :return: None
        """
        psutil.disp_normalmap(normal=self.N, height=self.height, width=self.width, delay=delay)

    def save_normalmap(self, filename=None):
        """
        Saves normal map as numpy array format (npy)
        :param filename: filename of a normal map
        :return: None
        """
        psutil.save_normalmap_as_npy(filename=filename, normal=self.N, height=self.height, width=self.width)

    def solve(self):
        if self.M is None:
            raise ValueError("Measurement M is None")
        if self.L is None:
            raise ValueError("Light L is None")
        if self.M.shape[1] != self.L.shape[1]:
            raise ValueError("Inconsistent dimensionality between M and L")

        #############################################

        # Please write your code here to solve the surface normal N whose size is (p, 3) as discussed in the tutorial

        # Step 1:

        N_estimated = np.linalg.lstsq(self.L.T, self.M.T, rcond=None)[0]  # 结果是 (3, p)

        # Step 2:
        self.N = normalize(N_estimated.T, axis=1)  # 归一化，使得每个法向量的模长为1

        #############################################

        if self.background_ind is not None:
            for i in range(self.N.shape[1]):
                self.N[self.background_ind, i] = 0

rps = None

def ps(action):

    global rps  # 引用全局变量 rps

    if action == "load image and light":
        # 加载图像和光源
        print("Loading image and light...")
        if rps is None:
            rps = PS()
        psutil.load_all(rps, mask_filename=MASK_FILENAME, light_filename=LIGHT_FILENAME, data_foldername=DATA_FOLDERNAME)
        print("Image and light loaded.")
    elif action == "run photometric stereo":
        # 运行光度立体法
        print("Running photometric stereo...")
        if rps is None:
            raise ValueError("rps not initialized. Please run 'load image and light' first.")
        psutil.solve_and_save(rps, normal_map_filename="./est_normal")
        print("Photometric stereo run completed.")
    elif action == "show results":
        # 显示结果
        print("Displaying results...")
        if rps is None:
            raise ValueError("rps not initialized. Please run 'load image and light' first.")
        psutil.evaluate_and_display(rps, GT_NORMAL_FILENAME)
        print("Results displayed.")
    else:
        print("Unknown action:", action)


