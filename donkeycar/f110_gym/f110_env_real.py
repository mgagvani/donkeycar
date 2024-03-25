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
import time

class f110_env(gym.Env):
    """
    OpenAI Gym Environment for a real car
    """

    STEER_LIMIT_LEFT = -1.0
    STEER_LIMIT_RIGHT = 1.0

    THROTTLE_MIN = 0.0
    THROTTLE_MAX = 1.0

    THROTTLE_CHANNEL = 0
    STEERING_CHANNEL = 1

    PCA9685_I2C_ADDR = 0x40
    PCA9685_I2C_BUSNUM = 7

    THROTTLE_FORWARD_PWM = 320 
    THROTTLE_STOPPED_PWM = 350 
    THROTTLE_REVERSE_PWM = 390

    STEERING_LEFT_PWM = 270        
    STEERING_RIGHT_PWM = 460

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
          outputs=['steering', 'throttle', 'user/mode', 'recording', 'web/buttons'],
          threaded=True)
        
        steering_controller = dk.parts.actuator.PCA9685(self.STEERING_CHANNEL, self.PCA9685_I2C_ADDR, busnum=self.PCA9685_I2C_BUSNUM)
        steering = dk.parts.actuator.PWMSteering(controller=steering_controller,
                                        left_pulse=self.STEERING_LEFT_PWM,
                                        right_pulse=self.STEERING_RIGHT_PWM)

        throttle_controller = dk.parts.actuator.PCA9685(self.THROTTLE_CHANNEL, self.PCA9685_I2C_ADDR, busnum=self.PCA9685_I2C_BUSNUM)
        throttle = dk.parts.actuator.PWMThrottle(controller=throttle_controller,
                                        max_pulse=self.THROTTLE_FORWARD_PWM,
                                        zero_pulse=self.THROTTLE_STOPPED_PWM,
                                        min_pulse=self.THROTTLE_REVERSE_PWM)

        self.V.add(steering, inputs=['steering'], threaded=True)
        self.V.add(throttle, inputs=['throttle'], threaded=True)
        
        camera = dk.parts.oak_d.OakD(
            enable_rgb=True,
            enable_depth=True,
            device_id=None
        )
        self.V.add(camera, inputs=[],
              outputs=['cam/image_array', 'cam/depth_array'],
              threaded=True)
        
        warp_points = [(640, 480), (0, 480), (0, 0), (640, 0)]
        warp_dst_birdseye = [(640, 480), (0, 480), (315, 0), (325, 0)] 
        warp = ImgWarp((640, 480), (640, 480), warp_points, warp_dst_birdseye)
        # self.V.add(warp, inputs=['cam/image_array'], outputs=['cam/image_array'], threaded=False)

        # lidar = dk.parts.lidar.RPLidar(0, 360)
        # self.V.add(lidar, inputs=[],outputs=['lidar/dist_array'], threaded=True)

        # start the vehicle
        #

        self.steps_alive = 0
        
        
    def step(self, action):
        # update vehicle with new action
        steering, throttle = action
        self.V.mem['steering'] = steering
        self.V.mem['throttle'] = throttle  

        # add to replay buffer deque
        self.replay_buffer.append(action)  
        self.steps_alive += 1  

        # obs is 'cam/image_array' for now
        obs = self.V.mem['cam/image_array']

        # calculate birds eye view
        # birdseye = birds_eye_view(obs)
        # self.V.mem['cam/image_array'] = birdseye

        # reward is 0.1 for now
        reward = self.calc_reward()

        # done is False for now
        done = False

        # info is None for now
        info = None

        return obs, reward, done, info

    def calc_reward(self):
        return self.steps_alive
    
    def calc_episode_over(self):
        """
        Episode if over if
        1) Lidar too close to an object
        2) Camera doesn't see the track
        """        

    def reset(self):
        print("RESETTING, self.steps_alive = ", self.steps_alive)
        self.V.start(rate_hz=self.loop_speed)
        self.steps_alive = 0

        # replay buffer, but reverse the throttle (so it goes to where it started)
        for item in self.replay_buffer:
            item[1] = -item[1]

        # reset the vehicle
        for _ in range(self.steps_alive):
            self.V.mem['steering'], self.V.mem['throttle'] = self.replay_buffer.popleft()
            time.sleep(0.05)

    def render(self, mode='human'):
        # web controller should auto update w/ 'cam/image_array'
        pass

if __name__ == "__main__":
    print("HI")
    env = f110_env()
    env.reset()
    print("Successfully RESET")

    # test reset: go forwards/right for 10 steps
    for _ in range(10):
        action = (0, -0.5)
        obs, reward, done, info = env.step(action)
        time.sleep(0.05)
        print(_)

    env.reset()

