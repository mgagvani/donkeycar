import donkeycar as dk
from donkeycar.parts.controller import LocalWebController
from donkeycar.parts import actuator
from donkeycar.parts import oak_d
from donkeycar.parts import lidar

import pupil_apriltags as apriltag
import depthai
from simple_pid import PID
import cv2

STEER_LIMIT_LEFT = -1.0
STEER_LIMIT_RIGHT = 1.0

THROTTLE_MIN = 0.0
THROTTLE_MAX = 1.0

THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

PCA9685_I2C_ADDR = 0x40
PCA9685_I2C_BUSNUM = 7

THROTTLE_FORWARD_PWM = 300 # 320
THROTTLE_STOPPED_PWM = 350
THROTTLE_REVERSE_PWM = 410 # 390

STEERING_LEFT_PWM = 270 
STEERING_RIGHT_PWM = 460

class ApriltagDetector():
    '''
    Outputs throttle and steering to follow AprilTags
    '''
    steer_setpoint = 0
    steer_kP = -3.85

    throttle_setpoint = 0.2 # meters away
    throttle_kP = -0.8
    throttle_kI = -0.8

    def get_camera_params(self):
        with depthai.Device() as device:
            calibData = device.readCalibration()

            # intrinsics are 3x3 matrix
            intrinsics = calibData.getCameraIntrinsics(depthai.CameraBoardSocket.CENTER, 640, 480)
            fx, fy, cx, cy = intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]

        device.close()
        return fx, fy, cx, cy

    def __init__(self, tag_size=0.05, tag_family='tag36h11'):
        self.tag_size = tag_size
        self.tag_family = tag_family
        
        self.camera_params = self.get_camera_params()

        self.detector = apriltag.Detector(families=self.tag_family, nthreads=5)
        self.steerPID = PID(self.steer_kP, 0, 0, setpoint=self.steer_setpoint)
        self.throttlePID = PID(self.throttle_kP, self.throttle_kI, 0, setpoint=self.throttle_setpoint)

        # output limits
        self.steerPID.output_limits = (-1, 1)
        # 0.4 for wood/smooth surfaces
        # 0.55 for carpet 
        self.throttlePID.output_limits = (-0.4, 0.4) # for now

    def run(self, img_arr):
        '''
        Compute PID/throttle to follow setpoints, return
        If this doesn't work use kinematic bicycle model and apriltag is waypoint
        '''
        if img_arr is None:
            return 0, 0

        # convert to grayscale (1 channel) 
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        gray = cv2.split(gray)[0] # get first channel (grayscale 
        
        tags = self.detector.detect(gray, camera_params=self.camera_params, estimate_tag_pose=True, tag_size=self.tag_size)
        if len(tags) == 0:
            self.throttlePID.reset() # reset integral term
            return 0, 0 # steer, throttle: stop
        else:
            tag = tags[0]
            # +1 is manual correciton
            tagX, tagZ = tag.pose_t[0], tag.pose_t[2]
            steer = self.steerPID(tagX)
            if type(steer) not in [int, float]:
                steer = steer[0]
            throttle = self.throttlePID(tagZ)
            # print(type(steer), type(throttle))
            print(f"steer: {steer}, throttle: {throttle}, tagX: {tagX}, tagZ: {tagZ}")
            return steer, throttle # testing

    def shutdown(self):
        pass
        


if __name__ == "__main__":
    V = dk.vehicle.Vehicle()

    controller = LocalWebController()

    V.add(controller,
          inputs=["cam/image_array", 'tub/num_records', 'user/mode', 'recording'],
          outputs=['steering', 'throttle', 'user/mode', 'recording', 'web/buttons'],
          threaded=True)

    V.add(ApriltagDetector(), 
          inputs=['cam/image_array'], 
          outputs=['pilot/steering', 'pilot/throttle'], 
          threaded=False)
    
    steering_controller = dk.parts.actuator.PCA9685(STEERING_CHANNEL, PCA9685_I2C_ADDR, busnum=PCA9685_I2C_BUSNUM)
    steering = dk.parts.actuator.PWMSteering(controller=steering_controller,
                                        left_pulse=STEERING_LEFT_PWM,
                                        right_pulse=STEERING_RIGHT_PWM)

    throttle_controller = dk.parts.actuator.PCA9685(THROTTLE_CHANNEL, PCA9685_I2C_ADDR, busnum=PCA9685_I2C_BUSNUM)
    throttle = dk.parts.actuator.PWMThrottle(controller=throttle_controller,
                                    max_pulse=THROTTLE_FORWARD_PWM,
                                    zero_pulse=THROTTLE_STOPPED_PWM,
                                    min_pulse=THROTTLE_REVERSE_PWM)
    
    V.add(steering, inputs=['pilot/steering'], threaded=True)
    V.add(throttle, inputs=['pilot/throttle'], threaded=True)
        
    camera = dk.parts.oak_d.OakD(
            enable_rgb=True,
            enable_depth=True,
            device_id=None
        )
    V.add(camera, inputs=[],
              outputs=['cam/image_array', 'cam/depth_array'],
              threaded=True)
    
    V.start(rate_hz=50)