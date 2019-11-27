# -*- coding: utf-8 -*-
""" python implementation of Face relighting.

Paper: 
    portrait lighting transfer using a mass transport approach. (2018)
Author's Matlab Implementation:
    https://github.com/AjayNandoriya/PortraitLightingTransferMTP
"""
import os
import sys
sys.path.insert(0, './PRNet')

import cv2
import numpy as np

from PRNet.api import PRN
from python_color_transfer.color_transfer import ColorTransfer

def _normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2) + 1e-6
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr


class Relight:
    """Methods for relighting human faces."""
    def __init__(self):
        self.prn = PRN(is_dlib=True, prefix='PRNet') 
        self.ct = ColorTransfer()
        self.mask1d = self.prn.face_ind
    def get_faces(self):
        '''get faces of 3d meshes, each face is a triplet of vertice inds.'''
        faces = self.prn.triangles
        return faces
    def get_normals(self, vertices=None):
        '''get normals of vertices. 
        
        Args:
            vertices: coordinates of vertices, shape=(n, 3)
            faces: faces represented by vertice ind triplet, shape=(m, 3)
        Returns:
            normals: normal vectors of vertices, shape=(n, 3)
        '''
        # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
        normals = np.zeros(vertices.shape, dtype=vertices.dtype )
        # Create an indexed view into the vertex array using the array of three indices for triangles
        faces = self.prn.triangles
        tris = vertices[faces]
        # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, 
        # and v2-v0 in each triangle             
        n = np.cross(tris[::,1]-tris[::,0], tris[::,2]-tris[::,0])
        # n is now an array of normals per triangle. The length of each normal is dependent the vertices, 
        # we need to normalize these, so that our next step weights each normal equally.
        _normalize_v3(n)
        # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
        # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle, 
        # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
        # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
        normals[faces[:,0]] += n
        normals[faces[:,1]] += n
        normals[faces[:,2]] += n
        _normalize_v3(normals)
        return normals
    def get_pos(self, img_arr=None):
        pos = self.prn.process(img_arr)
        return pos


def demo():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # path
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    img_folder = os.path.join(cur_dir, 'imgs')
    img_path = os.path.join(img_folder, 'portrait_s1.jpg')
    ref_path = os.path.join(img_folder, 'portrait_r1.jpg')
    # cls init
    RT = Relight()
    # inputs
    img_arr = cv2.imread(img_path)
    ref_arr = cv2.imread(ref_path)
    # get 3d positions by PRNet
    pos = RT.get_pos(img_arr=img_arr)
    ref_pos = RT.get_pos(img_arr=ref_arr)
    # obtain texture by remapping
    texture = cv2.remap(img_arr, pos[:,:,:2].astype(np.float32), 
                        None, interpolation=cv2.INTER_NEAREST, 
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    ref_texture = cv2.remap(ref_arr, pos[:,:,:2].astype(np.float32), 
                            None, interpolation=cv2.INTER_NEAREST, 
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    # colors, vertices, normals for color transfer
    colors = texture.reshape(-1, 3).astype(np.float64)
    ref_colors = ref_texture.reshape(-1, 3).astype(np.float64)
    vertices = pos[:,:,:2].reshape(-1, 2)
    ref_vertices = ref_pos[:,:,:2].reshape(-1, 2)
    normals = RT.get_normals(vertices=vertices)
    ref_normals = RT.get_normals(vertices=ref_vertices)
    features = np.concatenate((colors, vertices, normals), axis=1)
    ref_features = np.concatenate((ref_colors, ref_vertices, ref_normals), axis=1)
    assert features.shape == ref_features.shape
    assert features.dtype == ref_features.dtype
    # relighting by color transfer
    import pdb
    pdb.set_trace()

if __name__ == '__main__':
    demo()
