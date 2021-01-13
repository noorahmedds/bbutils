import numpy as np
from utils import *
import cv2

class BoundingBox:
    """
    Bounding Box Class
    """
    
    def __init__(self, xyxy):
        """Initialiser

        Args:
            coords (numpy array): 4 coordinated arbitrarily choses as x1,y1,x2,y2
        """

        self.xyxy = xyxy
        self.xcycwh = self.get_xcycwh()

        self.validate_bbox()

    def validate_bbox(self):
        """Assertion suite
        """
        # Evaluation set
        # Width and height cannot be negative or 0 (the current repo does not support inverted bounding boxes)
        # The bounding box should be an upright rectangle. This is automatically ensured with the xyxy notation

        assert self.xcycwh[2] > 0, print("The specified width is either negative or 0")
        assert self.xcycwh[3] > 0, print("The specified height is either negative or 0")
    
    def clamp_in_frame(self, frame_shape, in_place=True):
        """Clamps the bounding box in the frame

        Args:
            frame_shape (np.array): The direct output of the shape function
            in_place (bool, optional): Updates the coordinates in place otherwise returns the updates. Defaults to True.

        Returns:
            [type]: [description]
        """


        H,W,_ = frame_shape
        x1,y1,x2,y2 = self.xyxy

        x1 = clamp(x1, 0, W)
        y1 = clamp(y1, 0, H)
        x2 = clamp(x2, 0, W) 
        y2 = clamp(y2, 0, H)

        if in_place:
            self.xyxy = np.array([x1,y1,x2,y2])
            self.xcycwh = self.get_xcycwh()
            self.validate_bbox()

        return np.array([x1,y1,x2,y2])

        

    def is_overlapping(self, other_bbs, return_overlap_count = False):
        """Finds if the self bounding box is overlapping with arbitrary number of other bounding box 

        Args:
            other_bbs ([BoundingBox]): List of bounding boxes to check overlap with
            return_overlap_count (Bool): If asserted the function will return the number of boxes out of the list that the self bounding box was overlapping with

        Returns:
            overlap_flag (Bool): Asserted when current bounding box overlaps with all other_bbs
            overlap_count (int): Number of overlapping bounding boxes (returned only is return overlap count is asserted)
        """

        n_bbs = len(other_bbs)

        overlap_count = 0
        for bb in other_bbs: overlap_count += 1 if doOverlap(self.xyxy, bb.xyxy) else 0

        overlap_flag = overlap_count == n_bbs
            
        if return_overlap_count:
            return overlap_flag, overlap_count
        else:
            return overlap_flag

    
    def compute_iou(self, other_bb) -> float:
        """Computers Intersection Over union with another boundingBox

        Args:
            other_bb (BoundingBox)

        Returns:
            float: return value of intersection area over union area of the two bounding boxes
        """

        return intersection_over_union(self.xyxy, other_bb.xyxy)

    def get_xcycwh(self) -> np.ndarray:
        """Return converted bounding box coordinates

        Returns:
            np.ndarray: Convert Bounding box coordinates e.g. array([<xcenter>, <ycenter>, <width>, <height>])
        """

        x1,y1,x2,y2 = self.xyxy

        xc = ((x2-x1)/2) + x1
        yc = ((y2-y1)/2) + y1
        w = x2 - x1
        h = y2 - y1

        return np.array([xc,yc,w,h])

    def get_bottom_center(self) -> np.ndarray:
        """Quickly get the bottom center of the current bounding box. Usually used for projection of humans onto ground plane

        Returns:
            np.ndarray: A 2d point which is at the middle of the bottom line of the bounding box
        """

        x1,_,x2,y2 = self.xyxy
        return np.array([((x2 - x1) / 2) + x1, y2])

    def get_crop(self, frame:np.array, pad_flag=False): 
        """Returns a crop from a frame/image

        Args:
            frame (np.array): Frame read into a numpy array
            pad_flag (bool, optional): Pads the crop with appropriate padding if the bounding box coordinates exceed the frame. Defaults to False.

        Returns:
            np.ndarray: Cropped frame
        """

        x1,y1,x2,y2 = self.clamp_in_frame(frame.shape, in_place=False)
        crop = frame[y1:y2, x1:x2]

        if pad_flag:
            # First determine if any of the coords are outside the frame shape
            x1,y1,x2,y2 = self.xyxy
            H,W,_ = frame.shape

            left = right = top = bottom = 0

            if x1 < 0: left = x1
            if x2 > W: right = W - x2
            if y1 < 0: top = y1
            if y2 > H: bottom = H - y2

            crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT)

        return crop


def test_crop():
    frame = cv2.imread("./Screenshot from 2020-11-19 16-46-40.png")
    crop = crop_frame(frame, np.array([-100,-100,2000,2000]), pad_flag=True)
    
    cv2.imshow("Crop", crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Run tests here
    # test()
    pass