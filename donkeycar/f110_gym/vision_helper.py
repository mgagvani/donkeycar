import numpy as np
import cv2

class ImgWarp:
    '''
    ImgWarp: A part to warp an image to a birdseye view.
    '''

    def __init__(self, input_size, output_size, src_points, dst_points):
        '''
        __init__: initialize the ImgWarp part.
        input: input_size, a tuple (width, height)
        input: output_size, a tuple (width, height)
        input: src_points, a list of tuples [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        input: dst_points, a list of tuples [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.src_points = np.float32(src_points)
        self.dst_points = np.float32(dst_points)
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def run(self, img):
        '''
        run: warp an image to a birdseye view.
        input: img, an RGB numpy array
        output: the warped image
        '''
        # debug
        if img is None:
            return None

        return cv2.warpPerspective(img, self.M, (self.output_size[0], self.output_size[1]))
