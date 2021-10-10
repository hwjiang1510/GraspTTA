import os
import numpy as np
import torch
import mano
import json
import trimesh
import argparse
import mano

def fast_load_obj(file_obj, **kwargs):
    """
    Code slightly adapted from trimesh (https://github.com/mikedh/trimesh)
    Thanks to Michael Dawson-Haggerty for this great library !
    loads an ascii wavefront obj file_obj into kwargs
    for the trimesh constructor.
    vertices with the same position but different normals or uvs
    are split into multiple vertices.
    colors are discarded.
    parameters
    ----------
    file_obj : file object
                   containing a wavefront file
    returns
    ----------
    loaded : dict
                kwargs for trimesh constructor
    """
    # make sure text is utf-8 with only \n newlines
    text = file_obj.read()
    if hasattr(text, 'decode'):
        text = text.decode('utf-8')
    text = text.replace('\r\n', '\n').replace('\r', '\n') + ' \n'
    meshes = []
    def append_mesh():
        # append kwargs for a trimesh constructor
        # to our list of meshes
        if len(current['f']) > 0:
            # get vertices as clean numpy array
            vertices = np.array(
                current['v'], dtype=np.float64).reshape((-1, 3))
            # do the same for faces
            faces = np.array(current['f'], dtype=np.int64).reshape((-1, 3))
            # get keys and values of remap as numpy arrays
            # we are going to try to preserve the order as
            # much as possible by sorting by remap key
            keys, values = (np.array(list(remap.keys())),
                            np.array(list(remap.values())))
            # new order of vertices
            vert_order = values[keys.argsort()]
            # we need to mask to preserve index relationship
            # between faces and vertices
            face_order = np.zeros(len(vertices), dtype=np.int64)
            face_order[vert_order] = np.arange(len(vertices), dtype=np.int64)
            # apply the ordering and put into kwarg dict
            loaded = {
                'vertices': vertices[vert_order],
                'faces': face_order[faces],
                'metadata': {}
            }
            # build face groups information
            # faces didn't move around so we don't have to reindex
            if len(current['g']) > 0:
                face_groups = np.zeros(len(current['f']) // 3, dtype=np.int64)
                for idx, start_f in current['g']:
                    face_groups[start_f:] = idx
                loaded['metadata']['face_groups'] = face_groups
            # we're done, append the loaded mesh kwarg dict
            meshes.append(loaded)
    attribs = {k: [] for k in ['v']}
    current = {k: [] for k in ['v', 'f', 'g']}
    # remap vertex indexes {str key: int index}
    remap = {}
    next_idx = 0
    group_idx = 0
    for line in text.split("\n"):
        line_split = line.strip().split()
        if len(line_split) < 2:
            continue
        if line_split[0] in attribs:
            # v, vt, or vn
            # vertex, vertex texture, or vertex normal
            # only parse 3 values, ignore colors
            attribs[line_split[0]].append([float(x) for x in line_split[1:4]])
        elif line_split[0] == 'f':
            # a face
            ft = line_split[1:]
            if len(ft) == 4:
                # hasty triangulation of quad
                ft = [ft[0], ft[1], ft[2], ft[2], ft[3], ft[0]]
            for f in ft:
                # loop through each vertex reference of a face
                # we are reshaping later into (n,3)
                if f not in remap:
                    remap[f] = next_idx
                    next_idx += 1
                    # faces are "vertex index"/"vertex texture"/"vertex normal"
                    # you are allowed to leave a value blank, which .split
                    # will handle by nicely maintaining the index
                    f_split = f.split('/')
                    current['v'].append(attribs['v'][int(f_split[0]) - 1])
                current['f'].append(remap[f])
        elif line_split[0] == 'o':
            # defining a new object
            append_mesh()
            # reset current to empty lists
            current = {k: [] for k in current.keys()}
            remap = {}
            next_idx = 0
            group_idx = 0
        elif line_split[0] == 'g':
            # defining a new group
            group_idx += 1
            current['g'].append((group_idx, len(current['f']) // 3))
    if next_idx > 0:
        append_mesh()
    return meshes


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


def load_objects_HO3D(obj_root):
    object_names = ['011_banana', '021_bleach_cleanser', '003_cracker_box', '035_power_drill', '025_mug',
                    '006_mustard_bottle', '019_pitcher_base', '010_potted_meat_can', '037_scissors', '004_sugar_box']
    obj_pc, obj_face, obj_scale, obj_pc_resampled, obj_resampled_faceid = {}, {}, {}, {}, {}
    for obj_name in object_names:
        texture_path = os.path.join(obj_root, obj_name, 'textured_simple.obj')
        texture = fast_load_obj(open(texture_path))[0]
        obj_pc[obj_name] = texture['vertices']
        obj_face[obj_name] = texture['faces']
        obj_scale[obj_name] = get_diameter(texture['vertices'])
        obj_pc_resampled[obj_name] = np.load(texture_path.replace('textured_simple.obj', 'resampled.npy'))
        obj_resampled_faceid[obj_name] = np.load(texture_path.replace('textured_simple.obj', 'resample_face_id.npy'))
        #resample_obj_xyz(texture['vertices'], texture['faces'], texture_path)
    return obj_pc, obj_face, obj_scale, obj_pc_resampled, obj_resampled_faceid


def mapping_obj_name(obj_id):
    object_names = ['011_banana', '021_bleach_cleanser', '003_cracker_box', '035_power_drill', '025_mug',
                    '006_mustard_bottle', '019_pitcher_base', '010_potted_meat_can', '037_scissors', '004_sugar_box']
    name = [it for it in object_names if int(it[:3]) == obj_id]
    assert len(name) == 1
    return name[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_id", type=int, default=6)
    args = parser.parse_args()
    assert args.obj_id in [3, 4, 6, 10, 11, 19, 21, 25, 35, 37]

    obj_root = './models/HO3D_Object_models'  # root of the object models
    obj_pc_dict, obj_face_dict, obj_scale_dict, obj_pc_resample_dict, obj_resample_faceid_dict = load_objects_HO3D(obj_root)

    # load results, obj id are in line 139
    obj_id = args.obj_id
    diverse_grasp_result = json.load(open('./diverse_grasp/ho3d/obj_id_{}.json'.format(obj_id)))
    recon_params = diverse_grasp_result['recon_params']  # list of predicted MANO params
    R_list = diverse_grasp_result['R_list']          # list of Random Rotation MATRIX for rotating the object, in so(3)
    r_list = diverse_grasp_result['r_list']          # list of Random Rotation ANGLES for rotating the object, in R^(3)
    trans_list = diverse_grasp_result['trans_list']  # list of Random Translation for translating the object, in R^(3)

    # change the mano layer here
    with torch.no_grad():
        rh_mano = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                            model_type='mano',
                            use_pca=True,
                            num_pca_comps=45,
                            batch_size=1,
                            flat_hand_mean=True)
    rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3)  # [1, 1538, 3], face triangle indexes
    rh_mano.eval()
    meshes_all = []

    # load object mesh
    obj_name = mapping_obj_name(obj_id)
    obj_xyz_origin = np.array(obj_pc_dict[obj_name])
    obj_face_origin = np.array(obj_face_dict[obj_name])
    obj_mesh = trimesh.Trimesh(vertices=obj_xyz_origin, faces=obj_face_origin,
                               face_colors=[240,128,128])

    for i in range(len(recon_params)):  # 100 is the number of generated grasps
        recon_param = np.array(recon_params[i])
        recon_param = torch.tensor(recon_param, dtype=torch.float32)  # [1, 61]
        R = np.array(R_list[i])
        trans = np.array(trans_list[i])
        r = np.array(r_list[i])

        # hand mesh
        pred_mano = rh_mano(betas=recon_param[:, :10],
                            global_orient=recon_param[:, 10:13],
                            hand_pose=recon_param[:, 13:58],
                            transl=recon_param[:, 58:])
        hand_verts = pred_mano.vertices

        hand_verts = hand_verts.detach().squeeze(0).numpy()  # [778,3]
        recon_xyz = np.matmul(np.linalg.inv(R[:3, :3]), (hand_verts - R[:3, 3].reshape(1, -1)).T).T
        hand_mesh = trimesh.Trimesh(vertices=recon_xyz, faces=rh_faces.numpy().reshape((-1, 3)),
                                    face_colors=[int(0.85882353*255), int(0.74117647*255), int(0.65098039*255)])

        # vis current affordance
        meshes = [hand_mesh, obj_mesh]
        trimesh.Scene(meshes).show()
        meshes_all.append(hand_mesh)

    # vis all affordance
    meshes_all.append(obj_mesh)
    trimesh.Scene(meshes_all).show()
