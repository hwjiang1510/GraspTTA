import torch
import trimesh
from metric import contactutils


def load_batch_info(sample_info):
    obj_mesh = trimesh.Trimesh(vertices=sample_info["obj_verts"], faces=sample_info["obj_faces"])
    trimesh.repair.fix_normals(obj_mesh)

    obj_triangles = sample_info["obj_verts"][sample_info["obj_faces"]]
    exterior = contactutils.batch_mesh_contains_points(torch.from_numpy(sample_info["hand_verts"][None, :, :]).float(),
                                                       torch.from_numpy(obj_triangles)[None, :, :, :].float())
    penetr_mask = ~exterior.squeeze(dim=0)

    if penetr_mask.sum() == 0:
        max_depth = 0
    else:
        (result_close, result_distance, _, ) = trimesh.proximity.closest_point(obj_mesh, sample_info["hand_verts"][penetr_mask == 1])
        max_depth = result_distance.max()

    return max_depth