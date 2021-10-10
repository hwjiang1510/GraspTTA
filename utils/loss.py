import torch
import numpy as np
from pytorch3d.loss import chamfer_distance
from typing import Union
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds
from pytorch3d.structures import Meshes
from utils import utils_loss


def Contact_loss(obj_xyz, hand_xyz, cmap):
    '''
    # hand-centric loss, encouraging hand touching object surface
    :param obj_xyz: [B, N1, 3]
    :param hand_xyz: [B, N2, 3]
    :param cmap: [B, N1], dynamic possible contact regions on object
    :param hand_faces_index: [B, 1538, 3] hand index in [0, N2-1]
    :return:
    '''
    f1 = [697, 698, 699, 700, 712, 713, 714, 715, 737, 738, 739, 740, 741, 743, 744, 745, 746, 748, 749,
          750, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768]
    f2 = [46, 47, 48, 49, 164, 165, 166, 167, 194, 195, 223, 237, 238, 280, 281, 298, 301, 317, 320, 323, 324, 325, 326,
          327, 328, 329, 330, 331, 332, 333, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354,
          355]
    f3 = [356, 357, 358, 359, 375, 376, 386, 387, 396, 397, 402, 403, 413, 429, 433, 434, 435, 436, 437, 438,
          439, 440, 441, 442, 443, 444, 452, 453, 454, 455, 456, 459, 460, 461, 462, 463, 464, 465, 466, 467]
    f4 = [468, 469, 470, 471, 484, 485, 486, 496, 497, 506, 507, 513, 514, 524, 545, 546, 547, 548, 549,
          550, 551, 552, 553, 555, 563, 564, 565, 566, 567, 570, 572, 573, 574, 575, 576, 577, 578]
    f5 = [580, 581, 582, 583, 600, 601, 602, 614, 615, 624, 625, 630, 631, 641, 663, 664, 665, 666, 667,
          668, 670, 672, 680, 681, 682, 683, 684, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695]
    f0 = [73, 96, 98, 99, 772, 774, 775, 777]
    prior_idx = f1 + f2 + f3 + f4 + f5 + f0
    hand_xyz_prior = hand_xyz[:, prior_idx, :]  # only using prior points for contact map

    B = obj_xyz.size(0)

    obj_CD, _ = utils_loss.get_NN(obj_xyz, hand_xyz_prior)  # [B, N1] NN distance from obj pc to hand pc
    n_points = torch.sum(cmap)
    cmap_loss = obj_CD[cmap].sum() / (B * n_points)
    return 3000.0 * cmap_loss


def TTT_loss(hand_xyz, hand_face, obj_xyz, cmap_affordance, cmap_pointnet):
    '''
    :param hand_xyz:
    :param hand_face:
    :param obj_xyz:
    :param cmap_affordance: contact map calculated from predicted hand mesh
    :param cmap_pointnet: target contact map predicted from ContactNet
    :return:
    '''
    B = hand_xyz.size(0)

    # inter-penetration loss
    mesh = Meshes(verts=hand_xyz.cuda(), faces=hand_face.cuda())
    hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)
    nn_dist, nn_idx = utils_loss.get_NN(obj_xyz, hand_xyz)
    interior = utils_loss.get_interior(hand_normal, hand_xyz, obj_xyz, nn_idx).type(torch.bool)
    penetr_dist = 120 * nn_dist[interior].sum() / B  # batch reduction

    # cmap consistency loss
    consistency_loss = 0.0001 * torch.nn.functional.mse_loss(cmap_affordance, cmap_pointnet, reduction='none').sum() / B
    
    # hand-centric loss
    contact_loss = 2.5 * Contact_loss(obj_xyz, hand_xyz, cmap=nn_dist < 0.02**2)
    return penetr_dist, consistency_loss, contact_loss
