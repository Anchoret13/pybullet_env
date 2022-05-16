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


        self.cube_position = [-1.8, 0, 0]
        self.cube_orientation = p.getQuaternionFromEuler([0, 0, 0])

        self.collision_cube_position = [-1.8, 0, 0.3]

        self.ball_base_position = [0, -0.3, 0.3]

        self.ball_position = [-0.0, -0.15, 0.4]
        self.ball_orientation = p.getQuaternionFromEuler([0, 0, 0])
        '''
        self.collision_cube = p.loadURDF("./urdf/block/cube_0.3/cube2.urdf", 
                                    self.collision_cube_position, self.cube_orientation,
                                    useFixedBase=False,
                                    flags = p.URDF_USE_SELF_COLLISION)
        ''' 
        # self.cube = p.loadURDF("./urdf/block/cube_0.3/cube.urdf", 
        #                             self.cube_position, self.cube_orientation,
        #                             useFixedBase=False,
        #                             flags = p.URDF_USE_SELF_COLLISION)
        
        self.ball_base = p.loadURDF("./urdf/objects/table.urdf", 
                                    self.ball_base_position, self.cube_orientation,
                                    useFixedBase=True,
                                    flags = p.URDF_USE_SELF_COLLISION)
        
        self.ball = p.loadURDF("./urdf/ball_test.urdf", self.ball_position, self.ball_orientation)
        # For calculating the reward
        '''
        self.state_space = {
            'state' :,
            'achieve_goal' :,
            'desired_goal' :
        }
        '''
        self.state = self.get_state()

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    def uptown_funk(self, time = 120):
        # STOP! WAIT A MINUTE
        for _ in range(time):  
            self.step_simulation()

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
        throw: (x, y, z, roll, pitch, yaw, gripper_opening_length, release_time, car_velocity)
        move: (x, y, z, roll, pitch, yaw, gripper_opening_length, release_time, car_velocity)
        """
        assert control_method in ('end', 'joint')
        self.robot.move_ugv(action[-1])
        self.robot.move_ee(action[:-3], control_method)

        release_time = action[-2]
        self.uptown_funk(release_time)
        self.robot.move_gripper(action[-3])
        # reward = self.compute_reward(achieve_goal, goal)
        reward = self.update_reward()
        self.uptown_funk(400)

        done = True if reward == 1 else False
        info = dict(box_collide = self.box_collide)
        return self.get_state(), reward, done, info

    def update_reward(self):

        """
        realtime height check
        to be implemented
        """
        reward = -1
        if self.box_collide == True:
            print("SUCCESS!")
            reward = 0
        return reward

    def get_ball_obs(self):
        position, ori = p.getBasePosition(self.ball)
        # velocity = []
        velocity, angular_velocity = p.getBaseVelocity(self.ball)
        # return dict(position = position, velocity = velocity)
        return dict(ball_pos = position)

    def sample_goal(self):
        pass

    def get_state(self):
        state = dict()
        # ROBOT STATE
        robot_pos = self.robot.get_joint_obs()
        state.update(robot_pos = robot_pos)
        # ACHIEVED GOAL
        achieved_goal = self.get_ball_obs()
        state.update(achieved_goal)

        # DESIRED GOAL
        desired_goal = self.sample_goal()
        state.update(desired_goal)

        return state

    def get_observation(self):
        # USING THE CAMERA TO OBTAIN THE OBSERVATION
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_joint_obs())

        return obs

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.lianalg.norm(goal_a - goal_b, axis = 1)
    
    def compute_reward(self, achieved_goal, goal):
        d = self.goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d
        
    def reset_env(self):
        
        p.resetBasePositionAndOrientation(self.ball, self.ball_position, self.ball_orientation)
        # p.resetBasePositionAndOrientation(self.cube, self.cube_position, self.cube_orientation)
        # after initialize the position of the ball and cube, grasp the ball as initial state
        
        self.robot.move_ee((0, -0.15, 0.75, 1.570796251296997, 1.570796251296997, 1.570796251296997),'end')
        self.uptown_funk(120)
        self.robot.open_gripper()
        self.robot.move_ee((0, -0.15, 0.50, 1.570796251296997, 1.570796251296997, 1.570796251296997),'end')        
        self.uptown_funk(120)
        self.robot.close_gripper()
        self.uptown_funk(120)

        self.robot.move_ee((0, -0.15, 0.7, 1.570796251296997, 1.570796251296997, 1.570796251296997),'end')
        self.uptown_funk(120)

    def reset(self):
        self.robot.reset()
        self.reset_env()
        state = self.get_state()
        return state

    def close(self):
        p.disconnect(self.physicsClient)


import os
from robot import UR5Robotiq85, HuskyUR5
ycb_models = YCBModels(os.path.join('./data/ycb', '**', 'textured-decmp.obj'),)
camera = Camera((4, 0, 1),
                    (0, -0.7, 0),
                    (0, 0, 1),
                    0.1, 5, (320, 320), 40)
# robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))
robot = HuskyUR5((-0.2, 0.5, 0.5), (0, 0, 0))
env = Throwing(robot, ycb_models, camera, vis=True)
count = 0

env.reset()

while count < 10000:
    # env.step(env.read_debug_parameter(),'end')
    # env.step((-0.6, -0.1, 0.9, 1.570796251296997, 1.570796251296997, 1.570796251296997, 0.1),'end')
    env.step((-0.5, -0, 0.9, 1.570796251296997, 1.570796251296997, 1.570796251296997, 0.1, 40, -2))
    count = count + 1
    print(count)
    env.reset()

# Adjusting Initial State
# action :(x, y, z, row, pitch, yall, open_length)