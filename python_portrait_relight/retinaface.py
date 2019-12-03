# -*- coding: utf-8 -*-
""" RetinaFace Detection SDK. """

import torch
import numpy as np

from .pytorch_retinaface.data import cfg_mnet, cfg_re50
from .pytorch_retinaface.models.retinaface import RetinaFace
from .pytorch_retinaface.detect import load_model
from .pytorch_retinaface.layers.functions.prior_box import PriorBox
from .pytorch_retinaface.utils.box_utils import decode, decode_landm
from .pytorch_retinaface.utils.nms.py_cpu_nms import py_cpu_nms


class RetinaFaceSDK:
    """RetinaFace detection sdk.
    Attributes:
        cfg: configure params.
        device: cpu or cuda.
        detector: retina face detector.
        nms_thre: nms threshold.
    """
    def __init__(self, weight_path=None, cpu=None, cfg=None):
        """load model. """
        self.cfg = cfg
        net = RetinaFace(cfg=self.cfg, phase='test')
        net = load_model(net, weight_path, cpu)
        net.eval()
        self.device = torch.device("cpu" if cpu else "cuda")
        self.detector = net.to(self.device)
        self.nms_thre = 0.5
    def detect(self, img_arr=None, thre=None):
        """forward an img array.
        Args:
            img_arr: numpy array of an img returned by cv2.imread.
        Returns:
            detected boxes and landmarks.
        """
        img_arr, height, width = self.preprocess(img_arr)
        loc, conf, landms = self.detector(img_arr)  # forward pass
        boxes, landmarks, scores = self.postprocess((loc, conf, landms), (height, width),
                                                    resize=1, thre=thre)
        return [boxes, landmarks, scores]
    def preprocess(self, img_arr=None):
        """Preprocess. """
        img_arr = np.float32(img_arr)
        im_height, im_width, _ = img_arr.shape
        img_arr -= (104, 117, 123)
        img_arr = img_arr.transpose(2, 0, 1)
        img_arr = torch.from_numpy(img_arr).unsqueeze(0)
        img_arr = img_arr.to(self.device)
        return img_arr, im_height, im_width
    def postprocess(self, inputs, shape, resize=1, thre=None):
        """postprocess. """
        loc, conf, landms = inputs
        im_height, im_width = shape
        # scale
        scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(self.device)
        priors = PriorBox(self.cfg, image_size=(im_height, im_width)).forward().to(self.device)
        boxes = decode(loc.data.squeeze(0), priors.data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), priors.data, self.cfg['variance'])
        scale = torch.Tensor([im_width, im_height, im_width, im_height,
                              im_width, im_height, im_width, im_height,
                              im_width, im_height]).to(self.device)
        landms = landms * scale / resize
        landms = landms.cpu().numpy()
        # filter with score threshold
        if thre is not None:
            inds = np.where(scores > thre)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]
        # NMS
        inds = py_cpu_nms(np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False),
                          self.nms_thre)
        boxes = boxes[inds]
        scores = scores[inds]
        landms = landms[inds]
        return [boxes, landms, scores]
