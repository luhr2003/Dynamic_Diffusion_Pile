# Using new version

from collections import defaultdict
from copy import deepcopy
import os
import random
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
from utils import depth2fgpcd, downsample_pcd, fps, recenter

import ipdb
def obs2ptcl_fixed_num_batch(obs, particle_num, batch_size,action):
        assert type(obs) == np.ndarray
        assert obs.shape[-1] == 5
        assert obs[..., :3].max() <= 255.0
        assert obs[..., :3].min() >= 0.0
        assert obs[..., :3].max() >= 1.0
        assert obs[..., -1].max() >= 0.7 * 24
        assert obs[..., -1].max() <= 0.8 * 24
        

        depth = obs[..., -1] / 24
        mask=(depth<0.599/0.8).astype(np.int32)
        # mask=np.ones((self.screenHeight,self.screenWidth))
        # crop width >600
        mask[290:460,510:]=0
        rgb=obs[..., :3].astype(np.uint8)
        # set mask area to black
        rgb[mask==0]=0
        # show rgb
        batch_sampled_ptcl = np.zeros((batch_size, particle_num, 3))
        batch_particle_r = np.zeros((batch_size, ))
        batch_data=[]
        for i in range(batch_size):
            batch_sampled_ptcl=np.zeros((particle_num, 3))
            batch_particle_r=np.zeros((1,))
            fgpcd = depth2fgpcd(depth, mask, [869.1168308258057, 869.1168308258057, 360.0, 360.0])
            fgpcd = downsample_pcd(fgpcd, 0.01)
            sampled_ptcl, particle_r = fps(fgpcd, particle_num)
            batch_sampled_ptcl = recenter(fgpcd, sampled_ptcl, r = min(0.02, 0.5 * particle_r))
            batch_particle_r[0] = particle_r
            batch_data.append({'obs':batch_sampled_ptcl,'action':action})
        return batch_data

class PileDataset(torch.utils.data.Dataset):
    def __init__(self,path:str,obs_dims:int,fps_time:int,dynamics:bool=False):
        file_datas=[]
        if dynamics==False:
            for folder in os.listdir(path):
                file_name=os.path.join(path,folder,"info_dict_6.pt")
                file_datas.append(torch.load(file_name))
                
        self.final_data=[]
        self.sequence=defaultdict(list)
        for file_data in file_datas:
            self.raw_obs=file_data['raw_obs']
            self.action=file_data['action']
            for i in range(len(self.raw_obs)):
                # print(i)
                obs=self.raw_obs[i]
                action=self.action[i]
                action=np.array(action)
                self.final_data.extend(obs2ptcl_fixed_num_batch(obs,obs_dims//3,fps_time,action))
                self.sequence[i].append(action)
            
    
    def __len__(self):
        return len(self.final_data)
    
    def __getitem__(self,idx):
        return self.final_data[idx]
    
    def get_seq(self,id):
        return random.choice(self.sequence[id])

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
    path="dp_data/dp_data_12"
    env = FlexEnv(config,path)
    screenWidth = screenHeight = 720


    
    env.reset()
    img=pyflex.render(render_depth=True).reshape(720,720,5)
    # rgb=img[:,:,:3].astype(np.uint8)
    # plt.subplot(1,3,1)
    # plt.imshow(rgb)

    # random set position
    for j in range(200):
        pyflex.step()
        pyflex.render()
    
    # cur_position=env.get_positions().reshape(-1,4)
    # core_position=np.random.uniform(-5,5,2)
    # print(cur_position.shape)
    # num=cur_position.shape[0]
    # variance=2
    # rand_position=np.random.uniform((core_position-variance), (core_position+variance), (num,2))
    # print(rand_position.shape)
    # next_position=deepcopy(cur_position)
    # next_position[:,0]=rand_position[:,0]
    # next_position[:,2]=rand_position[:,1]
    # env.set_positions(next_position)


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

    action_seq_mpc_init = np.load('init_action/init_action_'+ str(n_sample) +'.npy')[np.newaxis, ...] # [1, 50, 4]
    action_label_seq_mpc_init = np.zeros(1)
    # subg_output = env.step_subgoal_ptcl(subgoal,
    #                                     GNN_single_model,
    #                                     None,
    #                                     n_mpc=n_mpc,
    #                                     n_look_ahead=n_look_ahead,
    #                                     n_sample=n_sample,
    #                                     n_update_iter=n_update_iter,
    #                                     mpc_type=mpc_type,
    #                                     gd_loop=gd_loop,
    #                                     particle_num=-1,
    #                                     funnel_dist=funnel_dist,
    #                                     action_seq_mpc_init=action_seq_mpc_init,
    #                                     action_label_seq_mpc_init=action_label_seq_mpc_init,
    #                                     time_lim=config['mpc']['time_lim'],
    #                                     auto_particle_r=True,)
    
    print("--------------start dp----------------")
    dataset=PileDataset("/home/isaac/dyn-res-pile-manip/dp_data",60,1,dynamics=False)
    print(dataset.sequence)
    for i in range(7):
        action=dataset.get_seq(i)
        action+=np.random.uniform(-0.3,0.3,4)
        env.step(action)
    





    
if __name__ == "__main__":
    main()
