'''this file is used to start the cross deform process'''
import argparse
import subprocess
import os
import sys

import tqdm 
import numpy as np

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_path', type=str,default="ours")
    id=0
    for x in np.arange(-5.5,2.5,0.3):
        for z in np.arange(-5.5,5.5,0.3):
            id+=1
            try:
                command="/home/isaac/anaconda3/envs/dyn-res-pile-manip/bin/python /home/isaac/dyn-res-pile-manip/heatmap.py --save_path {} --id {} --x {} --z {} --bolb {}".format("ours",id,x,z,0.6)
                command_list=command.split(" ")
                p=subprocess.Popen(command_list)
            except:
                print("error")
                continue