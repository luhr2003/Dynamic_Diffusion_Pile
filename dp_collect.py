# Using new version

import os
from typing import List
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
from teleop import teleoperation

# utils
from utils import load_yaml, save_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, set_seed, pcd2pix, gen_goal_shape, gen_subgoal, gt_rewards, gt_rewards_norm_by_sum, lighten_img, rmbg
from model.gnn_dyn import PropNetDiffDenModel

import ipdb

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

    path="dp_data/dp_data_9"
    os.makedirs(path, exist_ok=True)
    env = FlexEnv(config,path)
    screenWidth = screenHeight = 720


    
    env.reset()
    img=pyflex.render(render_depth=True).reshape(720,720,5)
    rgb=img[:,:,:3].astype(np.uint8)
    plt.subplot(1,3,1)
    # plt.imshow(rgb)

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

    funnel_dist = np.zeros_like(subgoal)

    for j in range(200):
        pyflex.step()
        pyflex.render()

    action_seq_mpc_init = np.load('init_action/init_action_'+ str(n_sample) +'.npy')[np.newaxis, ...] # [1, 50, 4]
    action_label_seq_mpc_init = np.zeros(1)
    print(env.get_cam_params())
    subg_output = env.step_subgoal_ptcl(subgoal,
                                        GNN_single_model,
                                        None,
                                        n_mpc=n_mpc,
                                        n_look_ahead=n_look_ahead,
                                        n_sample=n_sample,
                                        n_update_iter=n_update_iter,
                                        mpc_type=mpc_type,
                                        gd_loop=gd_loop,
                                        particle_num=-1,
                                        funnel_dist=funnel_dist,
                                        action_seq_mpc_init=action_seq_mpc_init,
                                        action_label_seq_mpc_init=action_label_seq_mpc_init,
                                        time_lim=config['mpc']['time_lim'],
                                        auto_particle_r=True,)
    
    torch.save(subg_output, os.path.join(path, "subg_output.pt"))
    id=0
    info_dict={"raw_obs":[],"state":[],"particle_r":[],"action":[]}
    for i in range(7):
        raw_obs,state,particle_r,action=teleoperation(env)
        info_dict["raw_obs"].append(raw_obs)
        info_dict["state"].append(state)
        info_dict["particle_r"].append(particle_r)
        info_dict["action"].append(action)
        torch.save(info_dict, os.path.join(path,"info_dict_"+str(id)+".pt"))
        id+=1
        # a=input("waiting for input:")
        # if a=="break":
        #     break
        
    torch.save(info_dict, os.path.join(path,"info_dict.pt"))
if __name__ == "__main__":
    main()
