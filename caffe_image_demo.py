import os
import cv2

import smoke_keypoint_caffe as skc

model_name = 'restiny_org_128x128x3'

sk = skc.SmokeKeypointCaffe(model_def='models/{}/deploy.prototxt'.format(model_name), model_weights='models/{}/deploy.caffemodel'.format(model_name))

imgpath = 'images/1635564856077.jpg'
image = cv2.imread(imgpath)

# get feat
ret = sk.predict(image, (0, 0, image.shape[1], image.shape[0]))

# post process
print(ret)