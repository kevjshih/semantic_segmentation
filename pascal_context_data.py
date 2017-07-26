"""
dataset encapsulation class for PASCAL Context Dataset
"""
import yaml
import scipy.io.loadmat as loadmat
import numpy as np

class PascalContextData(Data):
    def __init__(self, cfg):
        super().__init__()
        f = open(cfg, 'r')
        data = yaml.load(f)
        self._segdir = data.segdir
        self._VOCdir = data.VOCdir
        fid = open(data.idlist, 'r')
        fids = fid.readlines()       
        self._ids = [s.strip() for s in fids]
        self._batch_size = 10 # default
        f.close()

    def set_batch_size(bsize):
        self._batch_size = bsize
        
    def get_batch(ind):
        # implement me!

    def shuffle():
        np.random.shuffle(self_ids)

    def get_num_examples():
        return len(self._ids)

    def get_num_batches():
        # round up
        return round(len(self._ids) / self._batch_size)
