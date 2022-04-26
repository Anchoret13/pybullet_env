import time
import math
import random

import numpy as np
import pybullet as p
import pybullet_data

from utilities import Models, Camera, YCBModels
from collections import namedtuple
from attrdict import AttrDict
from tqdm import tqdm


class FailToReachTargetError(RuntimeError):
    pass


class Throwing:

    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, robot, models: Models, camera=None, vis=False) -> None:
        self.robot = robot
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.camera = camera

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.setTimeStep(self.SIMULATION_STEP_DELAY)
        self.planeID = p.loadURDF("plane.urdf")

        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # adjusting using sliders to tune parameters (name of the parameter,range,initial value)
        # self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
        # self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
        # self.zin = p.addUserDebugParameter("z", 0, 1., 0.5)
        # self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
        # self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi/2)
        # self.yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, np.pi/2)
        # self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.04)

        # initialize ee position
        self.xin = 0
        self.yin = 0
        self.zin = 0.4
        self.rollId = 0
        self.pitchId = np.pi/2
        self.yawId = np.pi/2
        self.gripper_opening_length_control = 0.04


        self.cube_position = [-1.5, 0.0, 0.0]
        self.cube_orientation = p.getQuaternionFromEuler([0, 0, 0])

        self.ball_position = [-0.0, 0.0, 0.0]
        self.ball_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.cube = p.loadURDF("/home/dyf/Desktop/robo/XWorld/games/xworld3d/models_3d/block/cube_0.7/cube.urdf", 
                                    self.cube_position, self.cube_orientation,
                                    useFixedBase=True)
        self.ball = p.loadURDF("./urdf/ball_test.urdf", self.ball_position, self.ball_orientation)

        # For calculating the reward
        self.box_collide = False

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    def read_debug_parameter(self):
        # FOR ENV DEBUGGING JUST IGNORE THIS
        x = p.readUserDebugParameter(self.xin)
        y = p.readUserDebugParameter(self.yin)
        z = p.readUserDebugParameter(self.zin)
        roll = p.readUserDebugParameter(self.rollId)
        pitch = p.readUserDebugParameter(self.pitchId)
        yaw = p.readUserDebugParameter(self.yawId)
        gripper_opening_length = p.readUserDebugParameter(self.gripper_opening_length_control)

        return x, y, z, roll, pitch, yaw, gripper_opening_length

    def step(self, action, control_method='end'):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for inverse kinematics
                         'joint' for forward kinematics
        """
        assert control_method in ('joint', 'end')
        self.robot.move_ee(action[:-1], control_method)

        for _ in range(40):  # Wait for a few steps
            self.step_simulation()
        self.robot.move_gripper(action[-1])
        for _ in range(120):  # Wait for a few steps
            self.step_simulation()

        reward = self.update_reward()
        done = True if reward == 1 else False
        info = dict(box_collide = self.box_collide)
        return self.get_observation(), reward, done, info

    def update_reward(self):
        reward = 0
        pass
        return reward

    def get_observation(self):
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_joint_obs())

        return obs


    def reset_env(self):
        # pass
        
        p.resetBasePositionAndOrientation(self.ball, self.ball_position, self.ball_orientation)
        p.resetBasePositionAndOrientation(self.cube, self.cube_position, self.cube_orientation)

        self.robot.move_ee((0, 0, 0.18, 1.570796251296997, 1.570796251296997, 1.570796251296997),'end')
        self.robot.move_gripper(0.02)
        for _ in range(120):  # Wait for a few steps
            self.step_simulation()

        self.robot.move_gripper(0.015)
        
        for _ in range(240):  # Wait for a few steps
            self.step_simulation()

        self.robot.move_ee((0, 0, 0.4, 1.570796251296997, 1.570796251296997, 1.570796251296997),'end')
        for _ in range(120):  # Wait for a few steps
            self.step_simulation()

        

    def reset(self):
        self.robot.reset()
        self.reset_env()
        return self.get_observation()

    def close(self):
        p.disconnect(self.physicsClient)

import os
from robot import UR5Robotiq85
ycb_models = YCBModels(os.path.join('./data/ycb', '**', 'textured-decmp.obj'),)
camera = Camera((4, 0, 1),
                    (0, -0.7, 0),
                    (0, 0, 1),
                    0.1, 5, (320, 320), 40)
robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))
env = Throwing(robot, ycb_models, camera, vis=True)
count = 0

env.reset()

while count < 10000:
    # env.step(env.read_debug_parameter(),'end')
    env.step((-0.6, -0.1, 0.9, 1.570796251296997, 1.570796251296997, 1.570796251296997, 0.1),'end')
# obs, reward, done, info = env.step((0, 0, 0.6, 1.570796251296997, 1.570796251296997, 1.570796251296997, 0.1),'end')
    count = count + 1
    print(count)

# Adjusting Initial State
# action :(x, y, z, row, pitch, yall, open_length)