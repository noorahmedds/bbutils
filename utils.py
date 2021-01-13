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

from shapely import Point

def get_centroid(xmin, ymin, xmax, ymax):
    cx = ((xmax - xmin) / 2) + xmin
    cy = ((ymax - ymin) / 2) + ymin
    return (cx, cy)

def update_entry_exit_stats_line(id_tracks, tracked_objects, momentum_term = (0.6, 0.9), distance_threshold = 15, n_average = 5, alt_logic = False, res = ()):
    """ Takes tracked objects and generates dictionary statistics for entry exit

    Args:
        tracked_objects ([type]): [description]
        momentum_term (float, optional): [description]. Defaults to 0.7.
        distance_threshold (int, optional): [description]. Defaults to 10.
        n_average (int, optional): [Is the number of centroids that are to be considered for averaging for thresholding]

    Updates the statistics inside stats_dict for entry exit. Ajju nice
    """

    for tracks in tracked_objects:
        identity = tracks.label
        if identity not in id_tracks:
            id_tracks[identity] = {
                "centroids":[],
                "bbox_area":[],
                "intersection": None,
                "available_previous_idx":0
            }

        xmin, ymin, xmax, ymax = tracks.rect
        
        if len(id_tracks[identity]["centroids"]) > 1:
            # Check if the two centroids cross intersect the exit line
            prev = id_tracks[identity]["centroids"][-2]

            # Add momentum to the centroids
            cx, cy = get_centroid(*tracks.rect)

            px = prev[0]
            py = prev[1]
            
            curr_area = (ymax-ymin) * (xmax-xmin)
            id_tracks[identity]["bbox_area"].append(curr_area)

            # Pick a momentum term between 0.5 and 0.7 based on the curr_area ratio of the bounding box. The larger the area the higher the momentum term and so on
            mt = momentum_term[0] + ((momentum_term[1] - momentum_term[0])*curr_area)

            cx = (mt * px) + ((1 - mt) * cx)
            cy = (mt * py) + ((1 - mt) * cy)
            centroid = (cx, cy)
            id_tracks[identity]["centroids"].append(centroid)
            
            threshold_flag = False
            if not alt_logic:
                # Caluclate distance from a previous point with which we meet a distanxe threshold and see if we are some threshold away. If we are we will continue the algo
                available_previous = id_tracks[identity]["centroids"][id_tracks[identity]["available_previous_idx"]]
                dist = np.linalg.norm(np.array(centroid) - np.array(available_previous))
                threshold_flag = dist > distance_threshold
            else:

                if n_average > len(id_tracks[identity]["centroids"]):
                    n_average = len(id_tracks[identity]["centroids"])
                centroid_distance_sums = 0
                prev = None
                for c, a in zip(id_tracks[identity]["centroids"][-1:-1*n_average:-1], id_tracks[identity]["bbox_area"][-1:-1*n_average:-1]):
                    if prev == None:
                        prev = c
                    else:
                        curr_dist = np.linalg.norm(np.array(prev) - np.array(c))
                        # The farther the person the larger this value. For small movements a larger distance is covered.
                        # To maintain pixel 
                        centroid_distance_sums +=  ((res[0]*res[1])/a) * curr_dist

                avg_distance_moved_per_frame = centroid_distance_sums/n_average
                threshold_flag = avg_distance_moved_per_frame > distance_threshold

        else:
            centroid = get_centroid(*tracks.rect)
            id_tracks[identity]["centroids"].append(centroid)

    return id_tracks