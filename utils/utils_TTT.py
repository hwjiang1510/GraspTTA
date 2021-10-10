import numpy as np
import torch

def aug_translation(obj_pc, origin_obj_pc, B):
    '''
    random translate object xyz, also apply same translation on origin obj xyz
    :param obj_pc: [1, 4, 3000]
    :return: translation augmented obj_xyz with batch B
    '''
    N1, N2 = obj_pc.size(2), origin_obj_pc.size(2)
    random_trans = 0.1 * (torch.rand((B, 3)) - 0.5)  # range [-0.05, 0.05], [B,3]
    random_trans2 = random_trans.clone()
    random_trans = random_trans.unsqueeze(2).repeat((1,1,N1))  # [B,3,N1]
    random_trans_origin = random_trans2.unsqueeze(2).repeat((1,1,N2))
    scale_trans = torch.zeros((B, 1, N1))
    scale_trans_origin = torch.zeros((B, 1, N2))
    trans = torch.cat((random_trans, scale_trans), dim=1)  # [B,4,N1]
    trans_origin = torch.cat((random_trans_origin, scale_trans_origin), dim=1)  # [B,4,N2]
    return obj_pc.repeat(B,1,1) + trans, origin_obj_pc.repeat(B,1,1) + trans_origin

def aug_translation_HO3D(obj_pc, B):
    '''
    random translate object xyz, also apply same translation on origin obj xyz
    :param obj_pc: [1, 4, 3000]
    :return: translation augmented obj_xyz with batch B
    '''
    N1 = obj_pc.size(2)
    random_trans = 0.1 * (torch.rand((B, 3)) - 0.5)  # range [-0.05, 0.05], [B,3]
    random_trans = random_trans.unsqueeze(2).repeat((1,1,N1))  # [B,3,N1]
    scale_trans = torch.zeros((B, 1, N1))
    trans = torch.cat((random_trans, scale_trans), dim=1).to(obj_pc.device)  # [B,4,N1]
    return obj_pc.repeat(B,1,1) + trans


