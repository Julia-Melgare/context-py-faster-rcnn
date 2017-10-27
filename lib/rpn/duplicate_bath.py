# --------------------------------------------------------
# Faster R-CNN + Places 365
# Copyright (c) 2017 PUCRS
# Written by Leandro Pereira da Silva
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps

DEBUG = False

class DuplicateBathLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
    	assert len(bottom) == 2,            'requires a single layer.bottom'
        assert len(top) == 1,               'requires a single layer.top'
        	
    def forward(self, bottom, top):
    	top[0].data[...] = bottom[1].data[...]
				
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        top[0].reshape(bottom[0].data.shape[0], bottom[1].data.shape[1])



