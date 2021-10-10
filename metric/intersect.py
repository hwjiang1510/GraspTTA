import json
import numpy as np
import trimesh
from joblib import Parallel, delayed


def intersect_vox(obj_mesh, hand_mesh, pitch=0.01):
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume


def intersect(obj_mesh, hand_mesh, engine="auto"):
    trimesh.repair.fix_normals(obj_mesh)
    inter_mesh = obj_mesh.intersection(hand_mesh, engine=engine)
    return inter_mesh


def get_all_volumes(sample_infos, save_results_path, workers=8):
    volumes = get_volumes_from_samples(sample_infos, workers=workers)
    volumes_clean = [volume for volume in volumes if volume is not None]
    skipped = len(volumes) - len(volumes_clean)
    with open(save_results_path, "w") as j_f:
        json.dump(
            {
                "mean_volume": np.mean(volumes_clean),
                "volumes": volumes_clean,
                "median_volume": np.median(volumes_clean),
                "std_volume": np.std(volumes_clean),
                "min_volume": np.min(volumes_clean),
                "max_volume": np.max(volumes_clean),
                "skipped": skipped,
                "computed": len(volumes_clean),
            },
            j_f,
        )
        print("Skipped {}, kept {}".format(skipped, len(volumes_clean)))


def get_volumes_from_samples(sample_infos, workers=8):
    volumes = Parallel(n_jobs=workers, verbose=5)(
        delayed(get_sample_intersect_volume)(sample_info)
        for sample_info in sample_infos
    )
    return volumes


def get_sample_intersect_volume(sample_info, mode="voxels"):
    hand_mesh = trimesh.Trimesh(vertices=sample_info["hand_verts"], faces=sample_info["hand_faces"])
    obj_mesh = trimesh.Trimesh(vertices=sample_info["obj_verts"], faces=sample_info["obj_faces"])
    if mode == "engines":
        try:
            intersection = intersect(obj_mesh, hand_mesh, engine="scad")
            if intersection.is_watertight:
                volume = intersection.volume
            else:
                intersection = intersect(obj_mesh, hand_mesh, engine="blender")
                if intersection.vertices.shape[0] == 0:
                    volume = 0
                elif intersection.is_watertight:
                    volume = intersection.volume
                else:
                    volume = None
        except Exception:
            intersection = intersect(obj_mesh, hand_mesh, engine="blender")
            if intersection.vertices.shape[0] == 0:
                volume = 0
            elif intersection.is_watertight:
                volume = intersection.volume
            else:
                volume = None
    elif mode == "voxels":
        volume = intersect_vox(obj_mesh, hand_mesh, pitch=0.005)
    return volume

