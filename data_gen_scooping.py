import os
import numpy as np
import pyflex
import time
import cv2


def quatFromAxisAngle(axis, angle):
    axis /= np.linalg.norm(axis)
    half = angle * 0.5
    w = np.cos(half)
    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two
    quat = np.array([axis[0], axis[1], axis[2], w])
    return quat

def init_multiview_camera(cam_dis = 3, cam_height = 4.5):
    camPos_list = []
    camAngle_list = []

    rad_list = np.deg2rad(np.array([0., 90., 180., 270.]) + 45.)
    cam_x_list = np.array([cam_dis, cam_dis, -cam_dis, -cam_dis])
    cam_z_list = np.array([cam_dis, -cam_dis, -cam_dis, cam_dis])

    for i in range(len(rad_list)):
        camPos_list.append(np.array([cam_x_list[i], cam_height, cam_z_list[i]]))
        camAngle_list.append(np.array([rad_list[i], -np.deg2rad(45.), 0.]))
    
    cam_intrinsic_params = np.zeros([len(camPos_list), 4]) # [fx, fy, cx, cy]
    cam_extrinsic_matrix = np.zeros([len(camPos_list), 4, 4]) # [R, t]
    
    return camPos_list, camAngle_list, cam_intrinsic_params, cam_extrinsic_matrix

def render(screenHeight, screenWidth, no_return=False):
    pyflex.step()
    if no_return:
        return
    else:
        return pyflex.render(render_depth=True).reshape(screenHeight, screenWidth, 5)

def data_gen_scooping():
    # info
    # debug = info['debug']
    # data_root_dir = info['data_root_dir']
    # headless = info['headless']
    
    # create folder
    # folder_dir = os.path.join(data_root_dir, 'granular_scooping')
    # os.system('mkdir -p ' + folder_dir)
    
    ## set scene
    pyflex.init(headless=False)
     
    radius = 0.03

    num_granular_ft = [5, 20, 5] 
    granular_scale = 0.25
    pos_granular = [0., 1., 0.]
    granular_dis = 0.

    draw_mesh = 0
    
    shapeCollisionMargin = 1e-100
    collisionDistance = 0.026
    dynamic_friction = 0.5
    granular_mass = 0.05

    # shape
    shape_type = 0 # 0: irreular shape; 1: regular shape
    shape_min_dist = 5. # 5. for irregular shape; 8 for regulra shape
    shape_max_dist = 10.
    
    scene_params = np.array([radius, *num_granular_ft, granular_scale, *pos_granular, granular_dis, 
                                    draw_mesh, shapeCollisionMargin, collisionDistance, dynamic_friction,
                                    granular_mass, shape_type, shape_min_dist, shape_max_dist])

    temp = np.array([0])
    pyflex.set_scene(35, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0)

    ## set env
    # add table
    table_height = 0.5
    halfEdge = np.array([4., table_height, 4.])
    center = np.array([0.0, 0.0, 0.0])
    quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
    hideShape = 0
    color = np.ones(3) * (160. / 255.)
    pyflex.add_box(halfEdge, center, quats, hideShape, color)
    table_shape_states = np.concatenate([center, center, quats, quats])

    # add bowl
    obj_shape_states = np.zeros((2, 14))
    bowl_scale = 30
    bowl_trans = np.array([0.5, table_height+0.8, 0.5])
    bowl_quat = quatFromAxisAngle(np.array([1., 0., 0.]), np.deg2rad(270.))
    bowl_color = np.array([204/255, 204/255, 1.])
    pyflex.add_mesh('bowl.obj', bowl_scale, 0, 
                    bowl_color, bowl_trans, bowl_quat, False)
    obj_shape_states[0] = np.concatenate([bowl_trans, bowl_trans, bowl_quat, bowl_quat])

    spoon_scale = 20
    spoon_trans = np.array([0.5, table_height+0.2, -2.0])
    spoon_quat_axis = np.array([1., 0., 0.])
    spoon_quat = quatFromAxisAngle(spoon_quat_axis, np.deg2rad(270.))
    spoon_color = np.array([204/255, 204/255, 1.])
    pyflex.add_mesh('spoon.obj', spoon_scale, 0,
                    spoon_color, spoon_trans, spoon_quat, False)
    # obj_shape_states[1] = np.concatenate([spoon_trans, spoon_trans, spoon_quat, spoon_quat])
    spoon_pos_prev = spoon_trans
    spoon_quat_prev = spoon_quat

    ## Light setting
    screebWidth, screenHeight = 720, 720
    pyflex.set_screenWidth(screebWidth)
    pyflex.set_screenHeight(screenHeight)
    pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
    pyflex.set_light_fov(70.)
    
    ## camera setting
    cam_dis = 6.
    cam_height = 10.
    camPos = np.array([cam_dis, cam_height, cam_dis])
    camAngle = np.array([np.deg2rad(45.), -np.deg2rad(45.), 0.])
    pyflex.set_camPos(camPos)
    pyflex.set_camAngle(camAngle)
    
    camPos_list, camAngle_list, cam_intrinsic_params, cam_extrinsic_matrix = init_multiview_camera(cam_dis, cam_height)

    pyflex.step()

    ## update the shape states for each time step
    lim_y = 3.
    lim_z = 0.4
    lim_x = 0.5
    lim_angle = 0.5
    count = 0
    for i in range(3000):
        n_stay_still = 40
        n_up = 200
        n_scoop = 1500
        
        if i < n_stay_still:
            angle_cur = 0.
            spoon_angle_delta = 0.
            spoon_pos_delta = np.zeros(3, dtype=np.float32)
        elif n_stay_still <= i < n_up:
            # spoon y position
            scale = 0.04
            spoon_pos_delta[1] = scale
            spoon_trans[1] += spoon_pos_delta[1]
            spoon_trans[1] = np.clip(spoon_trans[1], 0., lim_y)
            
            # spoon z position
            scale = 0.01
            spoon_pos_delta[2] = scale
            spoon_trans[2] += spoon_pos_delta[2]
            spoon_trans[2] = np.clip(spoon_trans[2], -2.0, lim_z)
            
            # spoon x position
            scale = 0.02
            spoon_pos_delta[0] = scale
            spoon_trans[0] -= spoon_pos_delta[0]
            spoon_trans[0] = np.clip(spoon_trans[0], -0.3, lim_x)
            
        elif n_up <= i < n_scoop:
            # spoon y position
            scale = 0.003 / 2
            spoon_pos_delta[1] = -scale
            spoon_trans[1] += spoon_pos_delta[1]
            spoon_trans[1] = np.clip(spoon_trans[1], table_height+0.8, lim_y)
            
            # spoon x position
            scale = 0.003 / 2
            spoon_pos_delta[0] = scale
            spoon_trans[0] += spoon_pos_delta[0]
            spoon_trans[0] = np.clip(spoon_trans[0], -0.3, 0.2)
            
            # spoon z position
            scale = 0.002 / 2
            spoon_pos_delta[2] = scale
            spoon_trans[2] -= spoon_pos_delta[2]
            spoon_trans[2] = np.clip(spoon_trans[2], 0.1, lim_z)
            
            # spoon angle
            scale = 0.004 / 2
            # spoon_angle_delta[2] = scale
            spoon_quat_axis += np.array([0., 0., scale])
            spoon_quat_axis[2] = np.clip(spoon_quat_axis[2], 0., lim_angle)
            spoon_quat = quatFromAxisAngle(spoon_quat_axis, np.deg2rad(270.))
            
        elif n_scoop <= i:
            # spoon y position
            scale = 0.003 / 2
            spoon_pos_delta[1] = scale
            spoon_trans[1] += spoon_pos_delta[1]
            spoon_trans[1] = np.clip(spoon_trans[1], table_height+0.4, lim_y)
            
            # spoon x position
            scale = 0.003 / 2
            spoon_pos_delta[0] = scale
            spoon_trans[0] += spoon_pos_delta[0]
            spoon_trans[0] = np.clip(spoon_trans[0], -0.3, 0.4)
            
            # spoon z position
            scale = 0.001 / 2
            spoon_pos_delta[2] = scale
            spoon_trans[2] += spoon_pos_delta[2]
            spoon_trans[2] = np.clip(spoon_trans[2], 0.1, lim_z-0.1)
            
            # spoon angle
            scale = 0.001 / 2
            # spoon_angle_delta[2] = scale
            spoon_quat_axis -= np.array([0., 0., scale])
            spoon_quat_axis[2] = np.clip(spoon_quat_axis[2], 0.3, lim_angle)
            spoon_quat = quatFromAxisAngle(spoon_quat_axis, np.deg2rad(270.))
            
        
        # set shape states
        shape_states = np.zeros((3, 14))
        shape_states[0] = table_shape_states
        
        # set shape state for table
        shape_states[0] = table_shape_states
        
        # set shape state for bowl
        shape_states[1] = obj_shape_states[0]
        
        # set shape state for spoon
        shape_states[2, :3] = spoon_trans
        shape_states[2, 3:6] = spoon_pos_prev
        shape_states[2, 6:10] = spoon_quat
        shape_states[2, 10:] = spoon_quat_prev
        
        spoon_pos_prev = spoon_trans
        spoon_quat_prev = spoon_quat
        
        pyflex.set_shape_states(shape_states)
        
        # if not debug and i % 2 == 0:
        #     for j in range(1):
        #         pyflex.set_camPos(camPos_list[j])
        #         pyflex.set_camAngle(camAngle_list[j])
                
        #         # create dir with cameras
        #         cam_dir = os.path.join(folder_dir, 'camera_%d' % (j))
        #         os.system('mkdir -p ' + cam_dir)
                
        #         # save rgb images
        #         img = render(screenHeight, screebWidth)
        #         cv2.imwrite(os.path.join(cam_dir, '%d_color.jpg' % count), img[:, :, :3][..., ::-1])
        
        #     count += 1
        
        pyflex.step()

    pyflex.clean()


if __name__ == "__main__":
    data_gen_scooping()