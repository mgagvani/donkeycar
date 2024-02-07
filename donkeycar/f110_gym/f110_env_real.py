import gymnasium as gym
import numpy as np

from gymnasium import error, spaces, utils
from collections import deque

import donkeycar as dk
from donkeycar.parts.controller import LocalWebController
from donkeycar.parts import actuator
from donkeycar.parts import oak_d
from donkeycar.parts import lidar

from vision_helper import *

class f110_env(gym.Env):
    """
    OpenAI Gym Environment for a real car
    """

    STEER_LIMIT_LEFT = -1.0
    STEER_LIMIT_RIGHT = 1.0

    THROTTLE_MIN = 0.0
    THROTTLE_MAX = 1.0

    VESC_SERIAL_PORT = '/dev/ttyACM0'
    VESC_MAX_SPEED_PERCENT = THROTTLE_MAX
    VESC_HAS_SENSOR = False
    VESC_START_HEARTBEAT = True
    VESC_BAUDRATE = 115200
    VESC_TIMEOUT = 0.1
    VESC_STEERING_SCALE = 1.0
    VESC_STEERING_OFFSET = 0.0

    def __init__(self, loop_speed=20):
        # gym settings
        self.action_space = spaces.Box(low=np.array([self.STEER_LIMIT_LEFT, self.THROTTLE_MIN]),
            high=np.array([self.STEER_LIMIT_RIGHT, self.THROTTLE_MAX]), dtype=np.float32 )
        
        # previous inputs
        self.replay_buffer = deque(maxlen=500)

        # time step
        self.loop_speed = loop_speed

        # vehicle and parts common to all observation spaces
        self.V = dk.vehicle.Vehicle()

        # controller, motors, camera, lidar, and other parts
        controller = LocalWebController()
        self.V.add(controller,
          inputs=["cam/image_array", 'tub/num_records', 'user/mode', 'recording'],
          outputs=['user/steering', 'user/throttle', 'user/mode', 'recording', 'web/buttons'],
          threaded=True)
        '''
        vesc = dk.parts.actuator.VESC(self.VESC_SERIAL_PORT,
                      self.VESC_MAX_SPEED_PERCENT,
                      self.VESC_HAS_SENSOR,
                      self.VESC_START_HEARTBEAT,
                      self.VESC_BAUDRATE,
                      self.VESC_TIMEOUT,
                      self.VESC_STEERING_SCALE,
                      self.VESC_STEERING_OFFSET
                    )
        self.V.add(vesc, inputs=['steering', 'throttle'])
        '''
        camera = dk.parts.oak_d.OakD(
            enable_rgb=True,
            enable_depth=True,
            device_id=None
        )
        self.V.add(camera, inputs=[],
              outputs=['cam/image_array', 'cam/depth_array'],
              threaded=True)

        lidar = dk.parts.lidar.RPLidar(0, 360)
        self.V.add(lidar, inputs=[],outputs=['lidar/dist_array'], threaded=True)

        # start the vehicle
        self.V.start(rate_hz=self.loop_speed)
        
        
    def step(self, action):
        # update vehicle with new action
        steering, throttle = action
        self.V.mem['user/steering'] = steering
        self.V.mem['user/throttle'] = throttle  

        # add to replay buffer deque
        self.replay_buffer.append(action)    

        # obs is 'cam/image_array' for now
        obs = self.V.mem['cam/image_array']

        # calculate birds eye view
        birdseye = birds_eye_view(obs)
        self.V.mem['cam/image_array'] = birdseye

        # reward is 0.1 for now
        reward = self.calc_reward()

        # done is False for now
        done = False

        # info is None for now
        info = None

        return obs, reward, done, info

    def calc_reward(self):
        return 0.1
    
    def calc_episode_over(self):
        """
        Episode if over if
        1) Lidar too close to an object
        2) Camera doesn't see the track
        """        

    def reset(self):
        pass

    def render(self, mode='human'):
        # web controller should auto update w/ 'cam/image_array'
        pass

if __name__ == "__main__":
    env = f110_env()
    env.reset()

    while True:
        action = (0.1, 0.1)
        obs, reward, done, info = env.step(action)

