"""
dataset encapsulation class for PASCAL Context Dataset
"""

import scipy.io.loadmat as loadmat
import numpy as np

class PascalContextData(Data):
    def __init__(self, cfg):
        super().__init__()

        cfg = yaml.load(f)
        self._segdir = cfg.segdir
        self._VOCdir = cfg.VOCdir
        fid = open(cfg.idlist, 'r')
        fids = fid.readlines()       
        self._ids = [s.strip() for s in fids]
        self._batch_size = 10 # default
        fid.close()

    def set_batch_size(self, bsize):
        self._batch_size = bsize
        
    def get_batch(self,ind):
        # implement me!

    def shuffle(self):
        np.random.shuffle(self_ids)

    def get_num_examples(self):
        return len(self._ids)

    def get_num_batches(self):
        # round up
        return round(len(self._ids) / self._batch_size)
