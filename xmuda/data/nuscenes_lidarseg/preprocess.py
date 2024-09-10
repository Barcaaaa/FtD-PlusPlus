import sys
sys.path.append("/data1/wuyao/code/BFtD-journal/")

import os
import os.path as osp
import numpy as np
import pickle

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.lidarseg.utils import LidarsegClassMapper

from xmuda.data.nuscenes_lidarseg.projection import map_pointcloud_to_image
from xmuda.data.nuscenes_lidarseg import splits

from nuscenes.utils.data_classes import LidarPointCloud

def preprocess(nusc, split_names, root_dir, out_dir,
               keyword=None, keyword_action=None, subset_name=None,
               location=None):
    # cannot process day/night and location at the same time
    assert not (bool(keyword) and bool(location))
    if keyword:
        assert keyword_action in ['filter', 'exclude']

    # init dict to save
    pkl_dict = {}
    for split_name in split_names:
         pkl_dict[split_name] = []

    # fine to coarse label mapping (only 16 classes out of 32 are actually used in NuScenes lidarseg)
    class_mapper = LidarsegClassMapper(nusc)
    fine_2_carse_mapping_dict = class_mapper.get_fine_idx_2_coarse_idx()
    fine_2_coarse_mapping = np.array(
        [fine_2_carse_mapping_dict[fine_idx] for fine_idx in range(len(fine_2_carse_mapping_dict))]
    )

    for i, sample in enumerate(nusc.sample):
        curr_scene_name = nusc.get('scene', sample['scene_token'])['name']

        # get if the current scene is in train, val or test
        curr_split = None
        for split_name in split_names:
            if curr_scene_name in getattr(splits, split_name):
                curr_split = split_name
                break
        if curr_split is None:
            continue

        if subset_name == 'night':
            if curr_split == 'train':
                if curr_scene_name in splits.val_night:
                    curr_split = 'val'
        if subset_name == 'singapore':
            if curr_split == 'train':
                if curr_scene_name in splits.val_singapore:
                    curr_split = 'val'
        if subset_name == 'all':
            if curr_split == 'train':
                if curr_scene_name in splits.val_all:
                    curr_split = 'val'

        # filter for day/night
        if keyword:
            scene_description = nusc.get("scene", sample["scene_token"])["description"]
            if keyword.lower() in scene_description.lower():
                if keyword_action == 'exclude':
                    # skip sample
                    continue
            else:
                if keyword_action == 'filter':
                    # skip sample
                    continue

        if location:
            scene = nusc.get("scene", sample["scene_token"])
            if location not in nusc.get("log", scene['log_token'])['location']:
                continue

        lidar_token = sample["data"]["LIDAR_TOP"]
        cam_front_token = sample["data"]["CAM_FRONT"]
        lidar_path, _, _ = nusc.get_sample_data(lidar_token)
        cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_front_token)

        print('{}/{} {} {}, current split: {}'.format(i + 1, len(nusc.sample), curr_scene_name, lidar_path, curr_split))

        sd_rec_lidar = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
        cs_record_lidar = nusc.get('calibrated_sensor',
                             sd_rec_lidar['calibrated_sensor_token'])
        pose_record_lidar = nusc.get('ego_pose', sd_rec_lidar['ego_pose_token'])
        sd_rec_cam = nusc.get('sample_data', sample['data']["CAM_FRONT"])
        cs_record_cam = nusc.get('calibrated_sensor',
                             sd_rec_cam['calibrated_sensor_token'])
        pose_record_cam = nusc.get('ego_pose', sd_rec_cam['ego_pose_token'])

        calib_infos = {
            "lidar2ego_translation": cs_record_lidar['translation'],
            "lidar2ego_rotation": cs_record_lidar['rotation'],
            "ego2global_translation_lidar": pose_record_lidar['translation'],
            "ego2global_rotation_lidar": pose_record_lidar['rotation'],
            "ego2global_translation_cam": pose_record_cam['translation'],
            "ego2global_rotation_cam": pose_record_cam['rotation'],
            "cam2ego_translation": cs_record_cam['translation'],
            "cam2ego_rotation": cs_record_cam['rotation'],
            "cam_intrinsic": cam_intrinsic,
        }

        # load lidar points
        # pts = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :3].T
        pc_original = LidarPointCloud.from_file(lidar_path)
        pc_ref = pc_original.points  # (4,N)
        pts = pc_ref[0:3, :]
        intensity = pc_ref[3:, :]

        # map point cloud into front camera image
        pts_valid_flag, pts_cam_coord, pts_img = map_pointcloud_to_image(pts, (900, 1600, 3), calib_infos)
        # fliplr so that indexing is row, col and not col, row
        pts_img = np.ascontiguousarray(np.fliplr(pts_img))  # 点云投影在图像上的位置索引

        # only use lidar points in the front camera image
        pts = pts[:, pts_valid_flag]  # 笛卡尔坐标系下的点云
        pts_cam_coord = pts_cam_coord[:, pts_valid_flag]  # 相机坐标系下的点云
        intensity = intensity[:, pts_valid_flag]
        # print(pts.shape)
        # print(intensity.shape)

        # load segmentation labels
        lidarseg_labels_filename = osp.join(nusc.dataroot, nusc.get('lidarseg', lidar_token)['filename'])
        seg_labels = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)  # [num_points]
        seg_labels = seg_labels[pts_valid_flag]  # discard labels of points outside of FoV
        seg_labels = fine_2_coarse_mapping[seg_labels]  # map from fine to coarse labels

        # convert to relative path
        lidar_path = lidar_path.replace(root_dir + '/', '')
        cam_path = cam_path.replace(root_dir + '/', '')

        # transpose to yield shape (num_points, 3)
        pts = pts.T
        pts_cam_coord = pts_cam_coord.T
        intensity = intensity.T

        # append data to train, val or test list in pkl_dict
        data_dict = {
            'points': pts,
            'pts_cam_coord': pts_cam_coord,
            'intensity': intensity,
            'seg_labels': seg_labels.astype(np.uint8),
            'points_img': pts_img,  # row, col format, shape: (num_points, 2)
            'lidar_path': lidar_path,
            'camera_path': cam_path,
            "sample_token": sample["token"],
            "scene_name": curr_scene_name,
            "calib": calib_infos
        }
        pkl_dict[curr_split].append(data_dict)

    # save to pickle file
    save_dir = osp.join(out_dir, 'preprocess')
    os.makedirs(save_dir, exist_ok=True)
    for split_name in split_names:
        save_path = osp.join(save_dir, '{}{}.pkl'.format(split_name, '_' + subset_name if subset_name else ''))
        with open(save_path, 'wb') as f:
            pickle.dump(pkl_dict[split_name], f)
            print('Wrote preprocessed data to ' + save_path)


if __name__ == '__main__':
    root_dir = '/data/xmw/nuscenes'
    out_dir = '/data/wuyao/dataset/nuscenes/preprocess_ssda'  # with depth and intensity info
    nusc = NuScenes(version='v1.0-trainval', dataroot=root_dir, verbose=True)
    # for faster debugging, the script can be run using the mini dataset
    # nusc = NuScenes(version='v1.0-mini', dataroot=root_dir, verbose=True)
    # We construct the splits by using the meta data of NuScenes:
    # USA/Singapore: We check if the location is Boston or Singapore.
    # Day/Night: We detect if "night" occurs in the scene description string.
    # preprocess(nusc, ['train', 'test'], root_dir, out_dir, location='boston', subset_name='usa')
    # preprocess(nusc, ['train', 'val', 'test'], root_dir, out_dir, location='singapore', subset_name='singapore')
    # preprocess(nusc, ['train', 'test'], root_dir, out_dir, keyword='night', keyword_action='exclude', subset_name='day')
    # preprocess(nusc, ['train', 'val', 'test'], root_dir, out_dir, keyword='night', keyword_action='filter', subset_name='night')

    # SSDA
    preprocess(nusc, ['train_singapore_labeled', 'train_singapore_unlabeled'], root_dir, out_dir)

    # SemanticKITTI/nuScenes-lidarseg (to evaluate against LiDAR transfer baseline)
    # preprocess(nusc, ['train', 'val', 'test'], root_dir, out_dir, subset_name='all')
