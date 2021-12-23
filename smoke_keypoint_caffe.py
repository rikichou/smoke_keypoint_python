import caffe
import cv2
import numpy as np

import onnx
import onnxruntime

class SmokeKeypointCaffe(object):
    def __init__(self,
        model_def='models/restiny_coco_128x128/deploy.prototxt',
        model_weights='models/restiny_coco_128x128/deploy.caffemodel'   
        ) -> None:
        """
        Fatigue CNN to CNN arch
        """        
        super().__init__()
        caffe.set_mode_cpu()

        # create caffe network
        self.model = caffe.Net(model_def, model_weights, caffe.TEST)

        self.image_size = (128, 128)
        self.mean = np.array([127, 127, 127])
        self.std = None

    def get_input_face(self, image, rect):
        sx,sy,ex,ey = rect
        h,w,c = image.shape

        sx = int(max(0, sx))
        sy = int(max(0, sy))
        ex = int(min(w-1, ex))
        ey = int(min(h-1, ey))

        return image[sy:ey, sx:ex, :],sx,sy,ex,ey

    def imnormalize(self, img, mean, std, to_rgb=True):
        """Inplace normalize an image with mean and std.

        Args:
            img (ndarray): Image to be normalized.
            mean (ndarray): The mean to be used for normalize.
            std (ndarray): The std to be used for normalize.
            to_rgb (bool): Whether to convert to rgb.

        Returns:
            ndarray: The normalized image.
        """
        # cv2 inplace normalization does not accept uint8
        assert img.dtype != np.uint8

        # check if convert to rgb
        if to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        
        # sub mean ?
        if not mean is None:
            mean = np.float64(mean.reshape(1, -1))
            cv2.subtract(img, mean, img)  # inplace
        
        # div std
        if not std is None:
            stdinv = 1 / std
            cv2.multiply(img, stdinv, img)  # inplace
        return img

    def preprocessing(self, image):
        # resize to [128, 128, 3]
        image = cv2.resize(image, self.image_size)

        # BGR-->GRAY [128, 128, 1]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # (img - mean) / std [128, 128]
        data = image.astype(np.float32)
        data = self.imnormalize(data, self.mean, self.std, to_rgb=False)
        # data = data[:,:,np.newaxis]
        
        # H,W,C --> C,H,W
        data = np.transpose(data, (2, 0, 1)).astype(np.float32)

        # new axis
        return data[np.newaxis, :, :, :]

    def decode(self, heatmaps):
        # K,H,W --> N,K,H,W
        heatmaps = heatmaps[np.newaxis,:,:,:]

        # get max score and location
        preds, maxvals = self._get_max_preds(heatmaps)

        return preds, maxvals

    def _get_max_preds(self, heatmaps):
        """Get keypoint predictions from score maps.

        Note:
            batch_size: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

        Returns:
            tuple: A tuple containing aggregated results.

            - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
            - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
        """
        assert isinstance(heatmaps,
                        np.ndarray), ('heatmaps should be numpy.ndarray')
        assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        N, K, _, W = heatmaps.shape
        heatmaps_reshaped = heatmaps.reshape((N, K, -1))
        idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
        maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
        preds[:, :, 0] = preds[:, :, 0] % W
        preds[:, :, 1] = preds[:, :, 1] // W

        preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
        return preds, maxvals        

    def predict(self, image, facerect):
        """
        Extract face features
        image: origin image
        facerect: sx,sy,ex,ey
        """
        # get input face
        faceimage,fsx,fsy,fex,fey = self.get_input_face(image, facerect)
        self.inputface_rect = [fsx,fsy,fex,fey]
        # preprocessing
        input_data = self.preprocessing(faceimage)

        # predict and get result
        self.model.blobs['input'].data[...] = input_data
        output = self.model.forward()

        return self.decode(np.array(output['output'][0]))