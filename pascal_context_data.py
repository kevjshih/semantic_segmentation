"""
dataset encapsulation class for PASCAL Context Dataset
"""

import scipy.io
from scipy import misc
import numpy as np
from data import Data
import os
import torch
class PascalContextData(Data):
	def __init__(self, cfg):
		super().__init__(cfg)
		self._segdir = cfg['segdir']
		self._VOCdir = cfg['VOCdir']
		fid = open(cfg['idlist'], 'r')
		fids = fid.readlines()		 
		self._ids = [s.strip() for s in fids]
		self._batch_size = 1 # default
		fid.close()

	def set_batch_size(self, bsize):
		self._batch_size = bsize
		
	def get_batch(self,ind):		
		curr_ids = self._ids[ind]

		
		id = curr_ids
		curr_im_path = self._VOCdir+'/JPEGImages/'+ id +'.jpg'
		curr_im = misc.imread(curr_im_path)
		curr_im = np.expand_dims(curr_im, 0)

		curr_label_path = self._segdir+'/trainval/'+id+'.mat'
		curr_label = scipy.io.loadmat(curr_label_path)
		curr_label = curr_label['LabelMap'] -1 # (set to 0-index)
		curr_label = np.expand_dims(curr_label, 0)
		
		curr_im = torch.Tensor(curr_im.astype(np.float32))
		curr_im = curr_im.permute(0,3,1,2)
		curr_label = torch.from_numpy(curr_label.astype(np.int64)).contiguous()
		return curr_im, curr_label
		
	def shuffle(self):
		np.random.shuffle(self._ids)

	def get_num_examples(self):
		return len(self._ids)

	def get_num_batches(self):
		# round up
		return round(len(self._ids) / self._batch_size)
