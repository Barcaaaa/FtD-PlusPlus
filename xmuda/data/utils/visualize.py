import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

from xmuda.data.utils.turbo_cmap import interpolate_or_clip, turbo_colormap_data


# all classes
NUSCENES_COLOR_PALETTE = [
    (255, 158, 0),  # car
    (255, 158, 0),  # truck
    (255, 158, 0),  # bus
    (255, 158, 0),  # trailer
    (255, 158, 0),  # construction_vehicle
    (0, 0, 230),  # pedestrian
    (255, 61, 99),  # motorcycle
    (255, 61, 99),  # bicycle
    (0, 0, 0),  # traffic_cone
    (0, 0, 0),  # barrier
    (200, 200, 200),  # background
]

# classes after merging (as used in xMUDA)
NUSCENES_COLOR_PALETTE_SHORT = [
    (255, 158, 0),  # vehicle
    (0, 0, 230),  # pedestrian
    (255, 61, 99),  # bike
    (0, 0, 0),  # traffic boundary
    (200, 200, 200),  # background
]

NUSCENES_LIDARSEG_COLOR_PALETTE_DICT = OrderedDict([
    ('ignore', (0, 0, 0)),  # Black
    ('barrier', (112, 128, 144)),  # Slategrey
    ('bicycle', (220, 20, 60)),  # Crimson
    ('bus', (255, 127, 80)),  # Coral
    ('car', (255, 158, 0)),  # Orange
    ('construction_vehicle', (233, 150, 70)),  # Darksalmon
    ('motorcycle', (255, 61, 99)),  # Red
    ('pedestrian', (0, 0, 230)),  # Blue
    ('traffic_cone', (47, 79, 79)),  # Darkslategrey
    ('trailer', (255, 140, 0)),  # Darkorange
    ('truck', (255, 99, 71)),  # Tomato
    ('driveable_surface', (0, 207, 191)),  # nuTonomy green
    ('other_flat', (175, 0, 75)),
    ('sidewalk', (75, 0, 75)),
    ('terrain', (112, 180, 60)),
    ('manmade', (222, 184, 135)),  # Burlywood
    ('vegetation', (0, 175, 0))  # Green
])

NUSCENES_LIDARSEG_COLOR_PALETTE = list(NUSCENES_LIDARSEG_COLOR_PALETTE_DICT.values())

NUSCENES_LIDARSEG_COLOR_PALETTE_SHORT = [
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['car'],  # vehicle
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['driveable_surface'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['sidewalk'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['terrain'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['manmade'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['vegetation'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['ignore']
]


# all classes
A2D2_COLOR_PALETTE_SHORT = [
    (255, 0, 0),  # car
    (255, 128, 0),  # truck
    (182, 89, 6),  # bike
    (204, 153, 255),  # person
    (255, 0, 255),  # road
    (150, 150, 200),  # parking
    (180, 150, 200),  # sidewalk
    (241, 230, 255),  # building
    (147, 253, 194),  # nature
    (255, 246, 143),  # other-objects
    (0, 0, 0)  # ignore
]

# colors as defined in https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
SEMANTIC_KITTI_ID_TO_BGR = {  # bgr
  0: [0, 0, 0],
  1: [0, 0, 255],
  10: [245, 150, 100],
  11: [245, 230, 100],
  13: [250, 80, 100],
  15: [150, 60, 30],
  16: [255, 0, 0],
  18: [180, 30, 80],
  20: [255, 0, 0],
  30: [30, 30, 255],
  31: [200, 40, 255],
  32: [90, 30, 150],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [75, 0, 175],
  50: [0, 200, 255],
  51: [50, 120, 255],
  52: [0, 150, 255],
  60: [170, 255, 150],
  70: [0, 175, 0],
  71: [0, 60, 135],
  72: [80, 240, 150],
  80: [150, 240, 255],
  81: [0, 0, 255],
  99: [255, 255, 50],
  252: [245, 150, 100],
  256: [255, 0, 0],
  253: [200, 40, 255],
  254: [30, 30, 255],
  255: [90, 30, 150],
  257: [250, 80, 100],
  258: [180, 30, 80],
  259: [255, 0, 0],
}
SEMANTIC_KITTI_COLOR_PALETTE = [SEMANTIC_KITTI_ID_TO_BGR[id] if id in SEMANTIC_KITTI_ID_TO_BGR.keys() else [0, 0, 0]
                                for id in range(list(SEMANTIC_KITTI_ID_TO_BGR.keys())[-1] + 1)]


# classes after merging (as used in xMUDA)
SEMANTIC_KITTI_COLOR_PALETTE_SHORT_BGR = [
    [245, 150, 100],  # car
    [180, 30, 80],  # truck
    [150, 60, 30],  # bike
    [30, 30, 255],  # person
    [255, 0, 255],  # road
    [255, 150, 255],  # parking
    [75, 0, 75],  # sidewalk
    [0, 200, 255],  # building
    [0, 175, 0],  # nature
    [255, 255, 50],  # other-objects
    [0, 0, 0],  # ignore
]
SEMANTIC_KITTI_COLOR_PALETTE_SHORT = [(c[2], c[1], c[0]) for c in SEMANTIC_KITTI_COLOR_PALETTE_SHORT_BGR]

VIRTUAL_KITTI_COLOR_PALETTE = [
    [0, 175, 0],  # vegetation_terrain
    [255, 200, 0],  # building
    [255, 0, 255],  # road
    [50, 255, 255],  # other-objects
    [80, 30, 180],  # truck
    [100, 150, 245],  # car
    [0, 0, 0],  # ignore
]

WAYMO_COLOR_PALETTE = [
    (200, 200, 200),  # unknown
    (255, 158, 0),  # vehicle
    (0, 0, 230),  # pedestrian
    (50, 255, 255),  # sign
    (255, 61, 99),  # cyclist
    (0, 0, 0),  # ignore
]

ERROR_COLOR_PALETTE = [
    [255, 0, 0],  # false
    [128, 128, 128],  # true
]

def show_raw_image(img, pic_idx):
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(pic_idx + '.png', bbox_inches='tight', pad_inches=0, quality=100, dpi=400)
    print(pic_idx + '.png')
    plt.cla()

def draw_points_image_labels(img, img_indices, seg_labels, pic_idx, show=False, color_palette_type='NuScenes', point_size=0.5):
    if color_palette_type == 'NuScenes':
        color_palette = NUSCENES_COLOR_PALETTE_SHORT
    elif color_palette_type == 'NuScenesLidarSeg':
        color_palette = NUSCENES_LIDARSEG_COLOR_PALETTE_SHORT
    elif color_palette_type == 'NuScenesLidarSegLong':
        color_palette = NUSCENES_LIDARSEG_COLOR_PALETTE
    elif color_palette_type == 'A2D2':
        color_palette = A2D2_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI_long':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE
    elif color_palette_type == 'VirtualKITTI':
        color_palette = VIRTUAL_KITTI_COLOR_PALETTE
    elif color_palette_type == 'Waymo':
        color_palette = WAYMO_COLOR_PALETTE
    else:
        raise NotImplementedError('Color palette type not supported')
    color_palette = np.array(color_palette) / 255.
    seg_labels[seg_labels == -100] = len(color_palette) - 1
    colors = color_palette[seg_labels]

    plt.imshow(img)
    plt.scatter(img_indices[:, 1], img_indices[:, 0], c=colors, alpha=0.5, s=point_size)
    plt.axis('off')

    plt.savefig(pic_idx + '.png', bbox_inches='tight', pad_inches=0, quality=100, dpi=400)
    print(pic_idx + '.png')
    # plt.tight_layout()
    if show:
        plt.show()
    plt.cla()


def normalize_depth(depth, d_min, d_max):
    # normalize linearly between d_min and d_max
    data = np.clip(depth, d_min, d_max)
    return (data - d_min) / (d_max - d_min)


def draw_points_image_depth(img, img_indices, depth, show=True, point_size=0.5):
    # depth = normalize_depth(depth, d_min=3., d_max=50.)
    depth = normalize_depth(depth, d_min=depth.min(), d_max=depth.max())
    colors = []
    for depth_val in depth:
        colors.append(interpolate_or_clip(colormap=turbo_colormap_data, x=depth_val))
    # ax5.imshow(np.full_like(img, 255))
    plt.imshow(img)
    plt.scatter(img_indices[:, 1], img_indices[:, 0], c=colors, alpha=0.5, s=point_size)

    plt.axis('off')

    if show:
        plt.show()


def draw_bird_eye_view(coords, seg_labels, pic_idx, color_palette_type, point_size, full_scale=4096):
    if color_palette_type == 'NuScenes':
        color_palette = NUSCENES_COLOR_PALETTE_SHORT
    elif color_palette_type == 'NuScenesLidarSeg':
        color_palette = NUSCENES_LIDARSEG_COLOR_PALETTE_SHORT
    elif color_palette_type == 'NuScenesLidarSegLong':
        color_palette = NUSCENES_LIDARSEG_COLOR_PALETTE
    elif color_palette_type == 'A2D2':
        color_palette = A2D2_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI_long':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE
    elif color_palette_type == 'VirtualKITTI':
        color_palette = VIRTUAL_KITTI_COLOR_PALETTE
    elif color_palette_type == 'Waymo':
        color_palette = WAYMO_COLOR_PALETTE
    else:
        raise NotImplementedError('Color palette type not supported')
    color_palette = np.array(color_palette) / 255.
    seg_labels[seg_labels == -100] = len(color_palette) - 1
    colors = color_palette[seg_labels]
    # print(seg_labels.shape)
    # print(colors.shape)

    plt.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.5, s=point_size)
    # plt.xlim([0, full_scale])
    # plt.ylim([0, full_scale])
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(pic_idx + '.png', bbox_inches='tight', pad_inches=0, quality=100, dpi=400)
    print(pic_idx + '.png')
    plt.cla()


def draw_bird_eye_view_error(coords, error_labels, pic_idx, point_size):
    color_palette = ERROR_COLOR_PALETTE
    color_palette = np.array(color_palette) / 255.
    colors = color_palette[error_labels]

    plt.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.5, s=point_size)
    # plt.xlim([0, full_scale])
    # plt.ylim([0, full_scale])
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(pic_idx + '.png', bbox_inches='tight', pad_inches=0, quality=100, dpi=400)
    print(pic_idx + '.png')
    plt.cla()
