"""
Class for batching datasets
Extend for each different dataset
"""

class Data:
	# initialize class using configuration object
	def __init__(self, cfg):
		pass
		
	# returns batch	corresponding to index
	def get_batch(ind):
		pass

	# partition the dataset into batches of size bsize
	def set_batch_size(bsize):
		pass

	# returns number of batches per epoch
	def get_num_batches():
		pass

	# returns number of examples in epoch
	def get_num_examples():
		pass
	
	# shuffles the dataset 
	def shuffle():
		pass
