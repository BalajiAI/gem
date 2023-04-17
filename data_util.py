import functools
import cv2
import numpy as np
import imageio
from glob import glob
import os
import shutil
import skimage
import pandas as pd
import json


def load_rgb(path, sidelength=None):
    img = imageio.imread(path)[:, :, :3]
    img = skimage.img_as_float32(img)

    img = square_crop_img(img)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_AREA)

    img -= 0.5
    img *= 2.
    img = img.transpose(2, 0, 1)
    return img


def load_params(filename):
    lines = open(filename).read().splitlines()

    params = np.array([float(x) for x in lines[0].split()]).astype(np.float32).squeeze()
    return params


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def square_crop_img(img):
    min_dim = np.amin(img.shape[:2])
    center_coord = np.array(img.shape[:2]) // 2
    img = img[center_coord[0] - min_dim // 2:center_coord[0] + min_dim // 2,
          center_coord[1] - min_dim // 2:center_coord[1] + min_dim // 2]
    return img


def train_val_split(object_dir, train_dir, val_dir):
    dirs = [os.path.join(object_dir, x) for x in ['pose', 'rgb', 'depth']]
    data_lists = [sorted(glob(os.path.join(dir, x)))
                  for dir, x in zip(dirs, ['*.txt', "*.png", "*.png"])]

    cond_mkdir(train_dir)
    cond_mkdir(val_dir)

    [cond_mkdir(os.path.join(train_dir, x)) for x in ['pose', 'rgb', 'depth']]
    [cond_mkdir(os.path.join(val_dir, x)) for x in ['pose', 'rgb', 'depth']]

    shutil.copy(os.path.join(object_dir, 'intrinsics.txt'), os.path.join(val_dir, 'intrinsics.txt'))
    shutil.copy(os.path.join(object_dir, 'intrinsics.txt'), os.path.join(train_dir, 'intrinsics.txt'))

    for data_name, data_ending, data_list in zip(['pose', 'rgb', 'depth'], ['.txt', '.png', '.png'], data_lists):
        val_counter = 0
        train_counter = 0
        for i, item in enumerate(data_list):
            if not i % 3:
                shutil.copy(item, os.path.join(train_dir, data_name, "%06d" % train_counter + data_ending))
                train_counter += 1
            else:
                shutil.copy(item, os.path.join(val_dir, data_name, "%06d" % val_counter + data_ending))
                val_counter += 1


def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def write_class_map(class_idxs, class_names, output_dir = "./"):
    # save out dictionary mapping class idx (in order fed to model) to class name
    # save using new order of indexes (i.e., what idx is saved as the label)
    class_map = {new_idx: class_names[orig_idx] for new_idx, orig_idx in enumerate(class_idxs)}
    print("writing out map")
   #if not os.path.exists(output_dir):os.makedirs(output_dir)
    with open(output_dir+"class_names.json", "w") as f:
        json.dump(class_map, f)


def read_class_map(output_dir):
    # read dictionary containing {idx: name} per class
    # idx is the idx relative to what is saved by the model as the label
    with open(output_dir+"class_names.json", "r") as f:
        return json.load(f)

def read_view_direction_rays(direction_file):
    img = cv2.imread(direction_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
    img -= 40000
    img /= 10000
    return img


def transform_viewpoint(v):
    """Transforms the viewpoint vector into a consistent representation"""

    return np.concatenate([v[:, :3],
                           np.cos(v[:, 3:4]),
                           np.sin(v[:, 3:4]),
                           np.cos(v[:, 4:5]),
                           np.sin(v[:, 4:5])], 1)


def euler2mat(z=0, y=0, x=0):
    Ms = []
    if z:
        cosz = np.cos(z)
        sinz = np.sin(z)
        Ms.append(np.array(
            [[cosz, -sinz, 0],
             [sinz, cosz, 0],
             [0, 0, 1]]))
    if y:
        cosy = np.cos(y)
        siny = np.sin(y)
        Ms.append(np.array(
            [[cosy, 0, siny],
             [0, 1, 0],
             [-siny, 0, cosy]]))
    if x:
        cosx = np.cos(x)
        sinx = np.sin(x)
        Ms.append(np.array(
            [[1, 0, 0],
             [0, cosx, -sinx],
             [0, sinx, cosx]]))
    if Ms:
        return functools.reduce(np.dot, Ms[::-1])
    return np.eye(3)


def look_at(vec_pos, vec_look_at):
    z = vec_look_at - vec_pos
    z = z / np.linalg.norm(z)

    x = np.cross(z, np.array([0., 1., 0.]))
    x = x / np.linalg.norm(x)

    y = np.cross(x, z)
    y = y / np.linalg.norm(y)

    view_mat = np.zeros((3, 3))

    view_mat[:3, 0] = x
    view_mat[:3, 1] = y
    view_mat[:3, 2] = z

    return view_mat
