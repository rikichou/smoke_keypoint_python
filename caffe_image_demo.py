import os
import cv2

import smoke_keypoint_caffe as skc

sk = skc.SmokeKeypointCaffe()

imgpath = 'images/1635564856077.jpg'
image = cv2.imread(imgpath)

# get feat
ret = sk.predict(image, (0, 0, image.shape[1], image.shape[0]))

# post process


print(ret)