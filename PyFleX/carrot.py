import os
from examples.data_gen_scooping import quatFromAxisAngle
import numpy as np
import pyflex
import cv2


# init pyflex
pyflex.init(headless=False)

# scene idx and params
scene_idx = 22

scene_param=[0.27741402, 0.27741402, -2.4882092, 0.5, -3.68636981, 1.0, 0.9, 1.0, 255.0, 10.0, 20.0, 16.0, 10.0, 16.0, 0.41612103, 0.0, -1.0, -1.0, -1.0, 0.0,0.075]
temp = np.array([0])

camPos=[0, 18, 0]
camAngle=[0, -1.57079633, 0]
pyflex.set_scene(22, scene_param, temp.astype(np.float64), temp, temp, temp, temp, 0)
pyflex.set_camPos(camPos)
pyflex.set_camAngle(camAngle)

table_height=0.5
obj_shape_states=np.zeros((2,14))
bowl_scale=30
bowl_trans=np.array([0.5,table_height+0.5,0.5])
bowl_quat=quatFromAxisAngle(np.array([1.,0.,0.]),np.deg2rad(270.))
bowl_color=np.array([204/255,204/255,1.])
pyflex.add_mesh("dustpan.obj",bowl_scale,0,bowl_color,bowl_trans,bowl_quat,False)

for j in range(500):
    pyflex.step()
    pyflex.render()