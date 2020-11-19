def doOverlap(boxA, boxB):
    if (boxA[0] > boxB[2] or boxB[0] > boxA[2]):
        return False
    if (boxA[1] > boxB[3] or boxB[1] > boxA[3]):
        return False

    return True

def intersection_over_union(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    intersection_area = (xB - xA + 1) * (yB - yA + 1)

    # Compute the area of both rectangles
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the IOU
    iou = float(intersection_area) / float(boxA_area + boxB_area - intersection_area)
    return iou


def clamp(value, low, high):
    if value < low:
        return low
    elif value > high:
        return high
    else:
        return value


import numpy as np
import cv2

def clamp_in_frame(frame_shape, xyxy):
    H,W,_ = frame_shape
    x1,y1,x2,y2 = xyxy

    x1 = clamp(x1, 0, W)
    y1 = clamp(y1, 0, H)
    x2 = clamp(x2, 0, W) 
    y2 = clamp(y2, 0, H)

    return np.array([x1,y1,x2,y2])

def crop_frame(frame:np.array, xyxy, pad_flag=False):
    x1,y1,x2,y2 = clamp_in_frame(frame.shape, xyxy)
    crop = frame[y1:y2, x1:x2]

    if pad_flag:
        # First determine if any of the coords are outside the frame shape
        x1,y1,x2,y2 = xyxy
        H,W,_ = frame.shape

        left = right = top = bottom = 0

        if x1 < 0: left = abs(x1)
        
        if x2 > W: right = abs(W - x2)
        
        if y1 < 0: top = abs(y1)
        
        if y2 > H: bottom = abs(H - y2)

        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT)

    return crop