#!/bin/bash
#coding = utf-8
apt-get -y install libusb-1.0-0
pip install pytorch fvcore chumpy open3d addict plyfile transforms3d opencv-python==3.4.2.17 numpy==1.16.2
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && git checkout 3b035f57f08295efc9af076ea60f62ad26d88b91
pip install -e . && cd ..
git clone https://github.com/hwjiang1510/MANO.git
cd MANO && pip install -e . && cd ..
pip install pybullet sk-video trimesh fvcore chumpy open3d addict plyfile transforms3d rtree opencv-python==3.4.2.17 numpy==1.16.2
python gen_diverse_grasp_ho3d.py --obj_id 3