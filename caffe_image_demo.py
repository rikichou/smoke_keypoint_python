import os
import cv2

import smoke_keypoint_caffe as skc

sk = skc.SmokeKeypointCaffe(model_def='models/restiny_128x128x3/deploy.prototxt', model_weights='models/restiny_128x128x3/deploy.caffemodel')

imgpath = 'images/1635564856077.jpg'
image = cv2.imread(imgpath)

# get feat
ret = sk.predict(image, (0, 0, image.shape[1], image.shape[0]))

# post process
print(ret)