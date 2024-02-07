import numpy as np
import cv2

def birds_eye_view(img):
    """
    Map image from front mounted camera to a birds eye view
    See the track as if you were looking down from above
    """

    # Define the perspective transform area
    src = np.float32([
        [0, img.shape[0]],
        [img.shape[1], img.shape[0]],
        [img.shape[1]*0.6, img.shape[0]*0.6],
        [img.shape[1]*0.4, img.shape[0]*0.6]
    ])

    dst = np.float32([
        [0, img.shape[0]],
        [img.shape[1], img.shape[0]],
        [img.shape[1], 0],
        [0, 0]
    ])

    # Get the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using the perspective transform matrix
    warped = cv2.warpPerspective(img, M, (640, 480), flags=cv2.INTER_LINEAR)

    return warped