import gymnasium as gym
import numpy as np

from gymnasium import error, spaces, utils
from collections import deque
from simple_pid import PID

import donkeycar as dk
from donkeycar.parts.controller import LocalWebController
from donkeycar.parts import actuator
from donkeycar.parts import oak_d
from donkeycar.parts import lidar

from vision_helper import *
from drivers import *
import time

def steering_conversion(steering_angle):
    '''
    convert steering angle from (-1, 1) to angle
    '''
    if steering_angle > 0:
        return 37.852 * steering_angle ** 2 \
                - 67.972 * steering_angle
    else:
        return -(37.852 * (-steering_angle) ** 2 \
                - 67.972 * (-steering_angle))
    

def angle_to_steering(angle):
    '''
    convert angle to steering angle
    https://www.desmos.com/calculator/ryhwmy5fcj
    '''
    if angle < -30.5: # 30.515
        return 1
    elif angle > 30.5:
        return -1

    # inverse of steering_conversion
    val = (67972 - np.sqrt(67972**2 + 151408000 * angle))/75704
    if angle < 0: # right
        val = (67972 - np.sqrt(67972**2 + 151408000 * angle))/75704
    else:
        val = -((67972 - np.sqrt(67972**2 + 151408000 * (-angle)))/75704)
    return val

class LidarConsumer:
    def __init__(self):
        self.backing_array = np.ones(360)
        self.quantized_array = np.zeros(30) # 30 points instead of 360
        self.driver = AnotherDriver()
        self.pid = PID(0.00032, 0.0001, 0.00000, setpoint=0) # diff between left/right distance
        # self.pid = PID(0.001, 0, 0, setpoint=500)

    def run(self, lidar_data):
        self.lidar_data = lidar_data
        # print(type(lidar_data), len(lidar_data))
        if len(lidar_data) == 0:
            return

        for item in lidar_data:
            # distance, angle
            distance, angle, _, _, _ = item
            # print(distance)
            
            nearest_angle = int(angle)
            self.backing_array[nearest_angle] = distance

        # quantize the array
        for i in range(0, 360, 12):
            self.quantized_array[i//12] = int(np.mean(self.backing_array[i:i+12]))

        # max dist and index (e.g angle)
        max_dist = np.max(self.backing_array)
        max_angle = np.argmax(self.backing_array)
        # print(f"{max_dist} @ {max_angle} deg", end='\r')

        max_quantized_dist = np.max(self.quantized_array)
        max_quantized_angle = np.argmax(self.quantized_array) * (360/30)

        speed, steering = self.driver.process_lidar(np.copy(self.backing_array))
        # print(f"Steering: {steering * 180/np.pi}", end='\r')
        
        # convert max angle to steering (-1, 1)
        steering = max_quantized_angle

        # left/right dist
        right_dist = np.mean(self.backing_array[45:135]) # 90
        left_dist = np.mean(self.backing_array[225:315]) # 270
        front_dist = np.mean(self.backing_array[0:45] + self.backing_array[315:360]) # 0-45, 315-360
        if front_dist < 2:
            return -0.85, 0
        elif front_dist > 5500:
            throttle = 0.95
        else:
            throttle = 0.9999999996

        print(f"Left: {left_dist}, Right: {right_dist}, Front: {front_dist}")

        steering = self.pid(left_dist - right_dist)
        # steering = self.pid(left_dist)
        print(f"Steering: {steering}")

        # scale angle to (-1, 1) <-- (-pi, pi)
        STEERING_FACTOR = 1.0
        # control_steer = angle_to_steering(steering)
        # # steering = -steering * STEERING_FACTOR / np.pi
        # print(f"Steering: {steering} --> {control_steer}", end='\r')
        return throttle, steering

    def shutdown(self):
        pass


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

    THROTTLE_FORWARD_PWM = 330 
    THROTTLE_STOPPED_PWM = 350 
    THROTTLE_REVERSE_PWM = 380

    STEERING_LEFT_PWM =  240 # 270        
    STEERING_RIGHT_PWM = 490 # 460

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
          outputs=['temp/steering', 'temp/throttle', 'user/mode', 'recording', 'web/buttons'],
          threaded=True)
          
        rplidar = lidar.RPLidar2(
            # min_angle=90,
            # max_angle=270,
            min_distance=100, # 100 mm = 10 cm
            # forward_angle=180,
            debug=True
        )
        # rplidar = lidar.RPLidar(90, 270, True)
        self.V.add(rplidar, inputs=[],outputs=['lidar/dist_array'], threaded=True)

        self.V.add(LidarConsumer(), inputs=['lidar/dist_array'], outputs=['throttle', 'steering'], threaded=False)
        
        camera = dk.parts.oak_d.OakD(
            enable_rgb=True,
            enable_depth=True,
            device_id=None
        )
        self.V.add(camera, inputs=[],
              outputs=['cam/image_array', 'cam/depth_array'],
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
        
        
        # warp_points = [(640, 480), (0, 480), (0, 0), (640, 0)]
        # warp_dst_birdseye = [(640, 480), (0, 480), (315, 0), (325, 0)] 
        # warp = ImgWarp((640, 480), (640, 480), warp_points, warp_dst_birdseye)
        # self.V.add(warp, inputs=['cam/image_array'], outputs=['cam/image_array'], threaded=False)

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
    input("Ready? ")
    print("HI")
    env = f110_env()
    env.reset()
    '''
    print("Successfully RESET")

    # test reset: go forwards/right for 10 steps
    for _ in range(10):
        action = (0, -0.5)
        obs, reward, done, info = env.step(action)
        time.sleep(0.05)
        print(_)

    env.reset()
    '''

