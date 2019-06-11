################################################################################################################################

import numpy as np

################################################################################################################################

# Straightforward IoU calculation that is not concerned with tensorflow formatting.

def calculate_iou(y_true, y_pred):
    # input must be as [x1, y1, x2, y2]
    
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    # AOG = Area of Groundtruth box
    AoG = np.abs(y_true[2] - y_true[0]) * np.abs(y_true[3] - y_true[1])

    # AOP = Area of Predicted box
    AoP = np.abs(y_pred[2] - y_pred[0]) * np.abs(y_pred[3] - y_pred[1])

    # overlaps are the co-ordinates of intersection box
    overlap_0 = np.maximum(y_true[0], y_pred[0])
    overlap_1 = np.maximum(y_true[1], y_pred[1])
    overlap_2 = np.minimum(y_true[2], y_pred[2])
    overlap_3 = np.minimum(y_true[3], y_pred[3])

    # intersection area
    intersection = np.abs(overlap_2 - overlap_0) * np.abs(overlap_3 - overlap_1)

    # area of union of both boxes
    union = AoG + AoP - intersection

    # iou calculation
    iou = intersection / union

    # return the mean IoU score for the batch
    return iou

################################################################################################################################
