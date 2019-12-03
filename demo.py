# -*- coding: utf-8 -*-
import os
import time

import cv2
import numpy as np

from relight import Relight

def demo():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # path
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    img_folder = os.path.join(cur_dir, 'imgs')
    img_num = 6
    img_names = ['portrait_s1.jpg' for _ in range(img_num)]
    ref_names = ['portrait_r{}.jpg'.format(i) for i in range(1, 1+img_num)]
    out_names = ['portrait_o{}.jpg'.format(i) for i in range(1, 1+img_num)]
    img_paths = [os.path.join(img_folder, x) for x in img_names]
    ref_paths = [os.path.join(img_folder, x) for x in ref_names]
    out_paths = [os.path.join(img_folder, x) for x in out_names]
    # cls init
    RT = Relight(c=8)
    for img_path, ref_path, out_path in zip(img_paths, ref_paths, out_paths):
        # inputs
        img_arr = cv2.imread(img_path)
        [h, w, c] = img_arr.shape
        print('{}: {}x{}x{}'.format(img_path, h, w, c))
        ref_arr = cv2.imread(ref_path)
        [h, w, c] = ref_arr.shape
        print('{}: {}x{}x{}'.format(ref_path, h, w, c))
        # relight
        stime = time.time()
        out_arr = RT.relight(img_arr=img_arr, ref_arr=ref_arr)
        print('relight time: {:.2f}\n'.format(time.time()-stime))
        # resize for display
        [ref_height, ref_width, _] = ref_arr.shape
        [height, width, _] = img_arr.shape
        ref_width = int(ref_width*height/ref_height)
        ref_height = height
        ref_arr = cv2.resize(ref_arr, (ref_width, ref_height))
        # save
        cv2.imwrite(out_path, np.concatenate((img_arr, ref_arr, out_arr), axis=1))

if __name__ == '__main__':
    demo()
