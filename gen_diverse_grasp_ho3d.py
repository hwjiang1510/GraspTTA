import os
import time
import torch
import argparse
from torch.utils.data import DataLoader
from dataset.HO3D_diversity_generation import HO3D_diversity
from network.affordanceNet_obman_mano_vertex import affordanceNet
from network.cmapnet_objhand import pointnet_reg
import numpy as np
import random
from utils import utils, utils_loss
import mano
import json
from utils.loss import TTT_loss
import trimesh
from metric.simulate import run_simulation


def intersect_vox(obj_mesh, hand_mesh, pitch=0.5):
    '''
    Evaluating intersection between hand and object
    :param pitch: voxel size
    :return: intersection volume
    '''
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume

def mesh_vert_int_exts(obj1_mesh, obj2_verts):
    inside = obj1_mesh.ray.contains_points(obj2_verts)
    sign = (inside.astype(int) * 2) - 1
    return sign


def main(args, model, cmap_model, eval_loader, device, rh_mano, rh_faces):
    '''
    Generate diverse grasps for object index with args.obj_id in out-of-domain HO3D object models
    '''
    model.eval()
    cmap_model.eval()
    rh_mano.eval()
    for batch_idx, (obj_id, obj_pc, origin_verts, origin_faces) in enumerate(eval_loader):
        if obj_id.item() != args.obj_id:
            continue
        obj_xyz = obj_pc.permute(0,2,1)[:,:,:3].squeeze(0).cpu().numpy()  # [3000, 3]
        origin_verts = origin_verts.squeeze(0).numpy()  # [N, 3]
        recon_params, R_list, trans_list, r_list = [], [], [], []

        for i in range(1000000):
            # generate random rotation
            rot_angles = np.random.random(3) * np.pi * 2
            theta_x, theta_y, theta_z = rot_angles[0], rot_angles[1], rot_angles[2]
            Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
            Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
            Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
            rot = Rx @ Ry @ Rz  # [3, 3]
            # generate random translation
            trans = np.array([-0.0793, 0.0208, -0.6924]) + np.random.random(3) * 0.2
            trans = trans.reshape((3, 1))
            R = np.hstack((rot, trans))  # [3, 4]
            obj_xyz_transformed = np.matmul(R[:3,0:3], obj_xyz.copy().T) + R[:3,3].reshape(-1,1)  # [3, 3000]
            obj_mesh_verts = (np.matmul(R[:3,0:3], origin_verts.copy().T) + R[:3,3].reshape(-1,1)).T  # [N, 3]
            obj_xyz_transformed = torch.tensor(obj_xyz_transformed, dtype=torch.float32)
            obj_pc_transformed = obj_pc.clone()
            obj_pc_transformed[0, :3, :] = obj_xyz_transformed  # [1, 4, N]

            obj_pc_TTT = obj_pc_transformed.detach().clone().to(device)
            recon_param = model.inference(obj_pc_TTT).detach()  # recon [1,61] mano params
            recon_param = torch.autograd.Variable(recon_param, requires_grad=True)
            optimizer = torch.optim.SGD([recon_param], lr=0.00000625, momentum=0.8)

            for j in range(300):  # non-learning based optimization steps
                optimizer.zero_grad()

                recon_mano = rh_mano(betas=recon_param[:, :10], global_orient=recon_param[:, 10:13],
                                     hand_pose=recon_param[:, 13:58], transl=recon_param[:, 58:])
                recon_xyz = recon_mano.vertices.to(device)  # [B,778,3], hand vertices

                # calculate cmap from current hand
                obj_nn_dist_affordance, _ = utils_loss.get_NN(obj_pc_TTT.permute(0, 2, 1)[:, :, :3], recon_xyz)
                cmap_affordance = utils.get_pseudo_cmap(obj_nn_dist_affordance)  # [B,3000]

                # predict target cmap by ContactNet
                recon_cmap = cmap_model(obj_pc_TTT[:, :3, :], recon_xyz.permute(0, 2, 1).contiguous())  # [B,3000]
                recon_cmap = (recon_cmap / torch.max(recon_cmap, dim=1)[0]).detach()

                penetr_loss, consistency_loss, contact_loss = TTT_loss(recon_xyz, rh_faces,
                                                                       obj_pc_TTT[:, :3, :].permute(0,2,1).contiguous(),
                                                                       cmap_affordance, recon_cmap)
                loss = 1 * contact_loss + 1 * consistency_loss + 7 * penetr_loss
                loss.backward()
                optimizer.step()
                if j == 0 or j == 299:
                    print("Object sample {}, pose {}, iter {}, "
                          "penetration loss {:9.5f}, "
                          "consistency loss {:9.5f}, "
                          "contact loss {:9.5f}".format(batch_idx, i, j,
                                                        penetr_loss.item(), consistency_loss.item(), contact_loss.item()))

            # evaluate grasp
            cam_extr = np.array([[1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., -1., 0.]]).astype(np.float32)
            obj_mesh_verts = obj_mesh_verts.dot(cam_extr[:3,:3].T)  # [N,3]
            obj_mesh = trimesh.Trimesh(vertices=obj_mesh_verts,
                                       faces=origin_faces.squeeze(0).cpu().numpy().astype(np.int32))  # obj
            final_mano = rh_mano(betas=recon_param[:, :10], global_orient=recon_param[:, 10:13],
                                 hand_pose=recon_param[:, 13:58], transl=recon_param[:, 58:])
            final_mano_verts = final_mano.vertices.squeeze(0).detach().cpu().numpy()  # [778, 3]
            final_mano_verts = final_mano_verts.dot(cam_extr[:3,:3].T)
            try:
                hand_mesh = trimesh.Trimesh(vertices=final_mano_verts, faces=rh_faces.cpu().numpy().reshape((-1, 3)))
            except:
                continue
            # penetration volume
            penetr_vol = intersect_vox(obj_mesh, hand_mesh, pitch=0.005)
            # contact
            penetration_tol = 0.005
            result_close, result_distance, _ = trimesh.proximity.closest_point(obj_mesh, final_mano_verts)
            sign = mesh_vert_int_exts(obj_mesh, final_mano_verts)
            nonzero = result_distance > penetration_tol
            exterior = [sign == -1][0] & nonzero
            contact = ~exterior
            sample_contact = contact.sum() > 0
            # simulation displacement
            vhacd_exe = "/hand-object/v-hacd/build/linux/test/testVHACD"
            try:
                simu_disp = run_simulation(final_mano_verts, rh_faces.reshape((-1, 3)),
                                          obj_mesh_verts, origin_faces.cpu().numpy().astype(np.int32).reshape((-1, 3)),
                                          vhacd_exe=vhacd_exe, sample_idx=i)
            except:
                simu_disp = 0.10
            save_flag = (penetr_vol < args.penetr_vol_thre) and (simu_disp < args.simu_disp_thre) and sample_contact
            print('generate id: {}, penetr vol: {}, simu disp: {}, contact: {}, save flag: {}'
                  .format(i, penetr_vol, simu_disp, sample_contact, save_flag))
            if save_flag:
                print('generate id {} saved'.format(i))
                recon_params.append(recon_param.detach().cpu().numpy().tolist())
                R_list.append(R.tolist())
                trans_list.append(trans.tolist())
                r_list.append(np.array([theta_x, theta_y, theta_z]).tolist())

            if len(r_list) == args.num_grasp:
                break

        save_path = './diverse_grasp/ho3d/obj_id_{}.json'.format(int(obj_id.item()))
        data = {
            'recon_params': recon_params,
            'R_list': R_list,
            'trans_list': trans_list,
            'r_list': r_list
        }
        with open(save_path, 'w') as f:
            json.dump(data, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''experiment setting'''
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_cuda", type=int, default=1)
    parser.add_argument("--dataloader_workers", type=int, default=32)
    '''affordance network information'''
    parser.add_argument("--affordance_model_path", type=str, default='checkpoints/model_affordance_best_full.pth')
    parser.add_argument("--encoder_layer_sizes", type=list, default=[1024, 512, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[1024, 256, 61])
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--obj_inchannel", type=int, default=4)
    parser.add_argument("--condition_size", type=int, default=1024)
    '''cmap network information'''
    parser.add_argument("--cmap_model_path", type=str, default='checkpoints/model_cmap_best.pth')
    '''Generated graps information'''
    parser.add_argument("--obj_id", type=int, default=6)
    # You can change the two thresholds to save the graps you want
    parser.add_argument("--penetr_vol_thre", type=float, default=4e-6)  # 4cm^3
    parser.add_argument("--simu_disp_thre", type=float, default=0.03)  # 3cm
    parser.add_argument("--num_grasp", type=int, default=100)  # number of grasps you want to generate
    args = parser.parse_args()
    assert args.obj_id in [3, 4, 6, 10, 11, 19, 21, 25, 35, 37]

    # device
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using device", device)

    # network
    affordance_model = affordanceNet(obj_inchannel=args.obj_inchannel,
                                     cvae_encoder_sizes=args.encoder_layer_sizes,
                                     cvae_latent_size=args.latent_size,
                                     cvae_decoder_sizes=args.decoder_layer_sizes,
                                     cvae_condition_size=args.condition_size)  # GraspCVAE
    cmap_model = pointnet_reg(with_rgb=False)  # ContactNet

    # load pre-trained model
    checkpoint_affordance = torch.load(args.affordance_model_path, map_location=torch.device('cpu'))['network']
    affordance_model.load_state_dict(checkpoint_affordance)
    affordance_model = affordance_model.to(device)
    checkpoint_cmap = torch.load(args.cmap_model_path, map_location=torch.device('cpu'))['network']
    cmap_model.load_state_dict(checkpoint_cmap)
    cmap_model = cmap_model.to(device)

    # dataset
    dataset = HO3D_diversity()
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
    # mano hand model
    with torch.no_grad():
        rh_mano = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                            model_type='mano',
                            use_pca=True,
                            num_pca_comps=45,
                            batch_size=1,
                            flat_hand_mean=True).to(device)
    rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).to(device)  # [1, 1538, 3], face indexes

    main(args, affordance_model, cmap_model, dataloader, device, rh_mano, rh_faces)

