import cv2
import numpy as np
import os
from glob import glob

def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow


if __name__ == "__main__":
    a = np.random.random(size=(3, 200, 200))
    b = np.random.random(size=(3, 200, 200))
    flow = compute_TVL1(a, b)
    print(flow.shape)





