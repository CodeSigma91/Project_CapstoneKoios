################################################################################################################################

import numpy      as np
import tensorflow as tf

################################################################################################################################

# This function is designed to work with Keras datagenerators as a model metric.

# SOURCE1: https://github.com/Balupurohit23/IOU-for-bounding-box-regression-in-Keras/blob/master/iou_metric.py
# SOURCE2: https://www.kaggle.com/vbookshelf/keras-iou-metric-implemented-without-tensor-drama

# I made use of the two sources because they each offered something I needed. I essentially used the framework of
# Source 2, but with the code of Source 1 because the format aligned with my data.

# I also reassigned the 'astype' numpy code to fit with tensorflow as well as converting the keras functions to np.

def tf_iou(y_true, y_pred):
    # iou as metric for bounding box regression
    # input must be as [x1, y1, x2, y2]
    
    results = []
    
    for i in range(0,y_true.shape[0]):
        
        y_true = tf.dtypes.cast(y_true, tf.float32)
        y_pred = tf.dtypes.cast(y_pred, tf.float32)

        # AOG = Area of Groundtruth box
        AoG = np.abs(np.transpose(y_true)[2] - np.transpose(y_true)[0] + 1) * np.abs(np.transpose(y_true)[3] -\
                                                                                     np.transpose(y_true)[1] + 1)

        # AOP = Area of Predicted box
        AoP = np.abs(np.transpose(y_pred)[2] - np.transpose(y_pred)[0] + 1) * np.abs(np.transpose(y_pred)[3] - \
                                                                                     np.transpose(y_pred)[1] + 1)

        # overlaps are the co-ordinates of intersection box
        overlap_0 = np.maximum(np.transpose(y_true)[0], np.transpose(y_pred)[0])
        overlap_1 = np.maximum(np.transpose(y_true)[1], np.transpose(y_pred)[1])
        overlap_2 = np.minimum(np.transpose(y_true)[2], np.transpose(y_pred)[2])
        overlap_3 = np.minimum(np.transpose(y_true)[3], np.transpose(y_pred)[3])

        # intersection area
        intersection = (overlap_2 - overlap_0 + 1) * (overlap_3 - overlap_1 + 1)

        # area of union of both boxes
        union = AoG + AoP - intersection

        # iou calculation
        iou = intersection / union

        # bounding values of iou to (0,1)
        epsilon = np.finfo(np.float32).eps
        iou     = np.clip(iou, 0.0 + epsilon, 1.0 - epsilon)
        iou     = tf.dtypes.cast(iou, tf.float32)
        
        # append the result to a list at the end of each loop
        results.append(iou)
    
    # return the mean IoU score for the batch
    return np.mean(results)

################################################################################################################################
