import trimesh
import os
import numpy as np
from utils import utils
import cv2

def load_objects_HO3D(obj_root):
    object_names = ['011_banana', '021_bleach_cleanser', '003_cracker_box', '035_power_drill', '025_mug',
                    '006_mustard_bottle', '019_pitcher_base', '010_potted_meat_can', '037_scissors', '004_sugar_box']
    obj_pc, obj_face, obj_scale, obj_pc_resampled, obj_resampled_faceid = {}, {}, {}, {}, {}
    for obj_name in object_names:
        texture_path = os.path.join(obj_root, obj_name, 'textured_simple.obj')
        texture = utils.fast_load_obj(open(texture_path))[0]
        obj_pc[obj_name] = texture['vertices']
        obj_face[obj_name] = texture['faces']
        obj_scale[obj_name] = get_diameter(texture['vertices'])
        obj_pc_resampled[obj_name] = np.load(texture_path.replace('textured_simple.obj', 'resampled.npy'))
        obj_resampled_faceid[obj_name] = np.load(texture_path.replace('textured_simple.obj', 'resample_face_id.npy'))
        #resample_obj_xyz(texture['vertices'], texture['faces'], texture_path)
    return obj_pc, obj_face, obj_scale, obj_pc_resampled, obj_resampled_faceid

def resample_obj_xyz(verts, faces, path):
    obj_mesh = trimesh.Trimesh(vertices=verts,
                               faces=faces)
    obj_xyz_resampled, face_id = trimesh.sample.sample_surface(obj_mesh, 3000)
    np.save(path.replace('textured_simple.obj', 'resampled.npy'), obj_xyz_resampled)
    np.save(path.replace('textured_simple.obj', 'resample_face_id.npy'), obj_xyz_resampled)

def get_diameter(vp):
    x = vp[:, 0].reshape((1, -1))
    y = vp[:, 1].reshape((1, -1))
    z = vp[:, 2].reshape((1, -1))
    x_max, x_min, y_max, y_min, z_max, z_min = np.max(x), np.min(x), np.max(y), np.min(y), np.max(z), np.min(z)
    diameter_x = abs(x_max - x_min)
    diameter_y = abs(y_max - y_min)
    diameter_z = abs(z_max - z_min)
    diameter = np.sqrt(diameter_x**2 + diameter_y**2 + diameter_z**2)
    return diameter

def readTxt(file_path):
    img_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip()
            img_list.append(item)
    file_to_read.close()
    return img_list

def pose_from_RT_HO3D(R, T):
    pose = np.zeros((4,4))
    pose[:3,3] = T
    pose[3,3] = 1
    R33, _ = cv2.Rodrigues(R)
    pose[:3, :3] = R33
    return pose

#_, _ = load_objects_HO3D('../models/HO3D_Object_models')