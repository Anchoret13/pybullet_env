import time
import math
import random
import os
import numpy as np
import pybullet as p
import pybullet_data
from robot import UR5Robotiq85
from utilities import YCBModels, Camera

from utilities import Models, Camera
from collections import namedtuple
from attrdict import AttrDict
from tqdm import tqdm



class FailToReachTargetError(RuntimeError):
    pass

class ThrowBall:
    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, robot, models: Models, camera = None, vis = False) -> None:
        self.robot = robot
        self.vis = vis
        if self.vis:
            self.p_bar =  tqdm(ncols=0, disable=False)
        self.camera = camera

        # define the env
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF("plane.urdf")

        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # custom slider to tune parameters
        self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
        self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
        self.zin = p.addUserDebugParameter("z", 0, 1., 0.5)
        self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
        self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi/2)
        self.yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, np.pi/2)
        self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.04)

        start_position = [-3.0, 0.0, 0.0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        ball_position = [-0.0, 0.0, 0.0]
        self.cube = p.loadURDF("/home/dyf/Desktop/robo/XWorld/games/xworld3d/models_3d/block/cube_0.7/cube.urdf", 
                                    start_position, start_orientation)
        self.ball = p.loadURDF("./urdf/ball_test.urdf", ball_position)

        # For calculating the reward
        self.box_contact = False

    def step_simulation(self):
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)
    
    def read_debug_parameter(self):
        # read the value of task parameter
        x = p.readUserDebugParameter(self.xin)
        y = p.readUserDebugParameter(self.yin)
        z = p.readUserDebugParameter(self.zin)
        roll = p.readUserDebugParameter(self.rollId)
        pitch = p.readUserDebugParameter(self.pitchId)
        yaw = p.readUserDebugParameter(self.yawId)
        gripper_opening_length = p.readUserDebugParameter(self.gripper_opening_length_control)

        return x, y, z, roll, pitch, yaw, gripper_opening_length

    def step(self, action, control_method='throw'):
        assert control_method in ('throw', 'move')
        self.robot.move_ee(action[:-1], control_method)
        self.robot.move_gripper(action[-1])
        for _ in range(120):  # Wait for a few steps
            self.step_simulation()
    
    def update_reward(self):
        reward = 0
        pass

    def get_observation(self):
        obs = dict
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb = rgb, depth = depth, seg = seg))
        else:
            assert self.camera is None
        
        obs.update(self.robot.get_joint_obs())

    def reset_env():
        # reset object position
        pass

    def reset(self):
        self.robot.reset()
        # self.reset_env()
        return self.get_observation()
