# Using new version

import os
import cv2
import pickle
import numpy as np
from env.flex_env import FlexEnv
import multiprocessing as mp
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib
from scipy.special import softmax
import pyflex
import cv2
from scipy.spatial.transform import Rotation

# utils
from utils import load_yaml, save_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, set_seed, pcd2pix, gen_goal_shape, gen_subgoal, gt_rewards, gt_rewards_norm_by_sum, lighten_img, rmbg
from model.gnn_dyn import PropNetDiffDenModel

import ipdb
    # 定义一个函数，用于捕获用户点击的点
def on_click(event, coords):
    ix, iy = event.xdata, event.ydata
    coords.append((ix, iy))
    print(f'Point clicked: ({ix}, {iy})')
    if len(coords) == 2:
        plt.close()


def pixel_to_world(pixel_coordinates, depth, camera_intrinsics, camera_extrinsics):
        # 将像素坐标点转换为相机坐标系
        camera_coordinates = np.dot(np.linalg.inv(camera_intrinsics), np.append(pixel_coordinates, 1.0))
        camera_coordinates *= depth

        # 将相机坐标系中的点转换为世界坐标系
        world_point = np.dot(np.linalg.inv(camera_extrinsics), np.append(camera_coordinates, 1.0))
        # world_point[2]=-world_point[2]
        return world_point[:3]


def teleoperation(env:FlexEnv):
    img=env.render().reshape(env.screenHeight,env.screenWidth,5)
    # 创建一个空列表，用于存储点击的点
    coords = []
    rgb=img[:,:,:3].astype(np.int32)
    depth=img[:,:,-1]
    depth=depth/env.global_scale

    # 显示图像并捕获点击事件
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    ax.set_title('Click on two points')
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, coords))


    plt.show()

    # 输出点击的点
    print(f'Clicked points: {coords}')
    coords=np.array(coords).astype(np.int32)

    camera_extrinsics=env.get_cam_extrinsics()
    camera_intrisics=env.get_cam_intrinsics()

    # 将像素坐标点转换为世界坐标系
    begin=pixel_to_world(np.array([coords[0][0],coords[0][1]]),depth[coords[0][1],coords[0][0]],camera_intrisics,camera_extrinsics)
    end=pixel_to_world(np.array([coords[1][0],coords[1][1]]),depth[coords[1][1],coords[1][0]],camera_intrisics,camera_extrinsics)
    begin=begin*env.global_scale
    end=end*env.global_scale
    print(f'begin:{begin},end:{end}')
    raw_obs,state,particle_r=env.get_process_observation()
    env.step([begin[0],begin[2],end[0],end[2]])
    action=[begin[0],begin[2],end[0],end[2]]
    return raw_obs,state,particle_r,action

def save_obs(env:FlexEnv):
    obs=env.render()
    assert type(obs) == np.ndarray
    assert obs.shape[-1] == 5
    assert obs[..., :3].max() <= 255.0
    assert obs[..., :3].min() >= 0.0
    assert obs[..., :3].max() >= 1.0
    assert obs[..., -1].max() >= 0.7 * env.global_scale
    assert obs[..., -1].max() <= 0.8 * env.global_scale
    

    depth = obs[..., -1] / env.global_scale
    rgb_mask=(depth<0.599/0.8).astype(np.int32)
    mask=(depth<0.599/0.8).astype(np.int32)
    # mask=np.ones((self.screenHeight,self.screenWidth))
    # crop width >600
    mask[290:460,510:]=0
    rgb=obs[..., :3].astype(np.uint8)
    # set mask area to black
    rgb[rgb_mask==0]=255
    # show rgb
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    plt.savefig('obs.png')
    plt.show()


def main():
    config = load_yaml("config/mpc/config.yaml")

    model_folder = config['mpc']['model_folder']
    model_iter = config['mpc']['iter_num']
    n_mpc = config['mpc']['n_mpc']
    n_look_ahead = config['mpc']['n_look_ahead']
    n_sample = config['mpc']['n_sample']
    n_update_iter = config['mpc']['n_update_iter']
    gd_loop = config['mpc']['gd_loop']
    mpc_type = config['mpc']['mpc_type']

    task_type = config['mpc']['task']['type']

    model_root = 'data/gnn_dyn_model/'
    model_folder = os.path.join(model_root, model_folder)
    GNN_single_model = PropNetDiffDenModel(config, True)
    if model_iter == -1:
        GNN_single_model.load_state_dict(torch.load(f'{model_folder}/net_best.pth'), strict=False)
    else:
        GNN_single_model.load_state_dict(torch.load(f'{model_folder}/net_epoch_0_iter_{model_iter}.pth'), strict=False)
    GNN_single_model = GNN_single_model.cuda()

    env = FlexEnv(config)
    screenWidth = screenHeight = 720

    if task_type == 'target_control':
        goal_row = config['mpc']['task']['goal_row']
        goal_col = config['mpc']['task']['goal_col']
        goal_r = config['mpc']['task']['goal_r']
        subgoal, mask = gen_subgoal(goal_row,
                                    goal_col,
                                    goal_r,
                                    h=screenHeight,
                                    w=screenWidth)
        goal_img = (mask[..., None]*255).repeat(3, axis=-1).astype(np.uint8)
    elif task_type == 'target_shape':
        goal_char = config['mpc']['task']['target_char']
        subgoal, goal_img = gen_goal_shape(goal_char,
                                            h=screenHeight,
                                            w=screenWidth)
    else:
        raise NotImplementedError
    
    env.reset()

    funnel_dist = np.zeros_like(subgoal)
    for j in range(200):
        pyflex.step()
        pyflex.render()
    while(1):
        teleoperation(env)
        save_obs(env)
    



    

if __name__ == "__main__":
    main()
