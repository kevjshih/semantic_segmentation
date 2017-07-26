"""
Class for batching datasets
Extend for each different dataset
"""

class Data:
	# initialize class using configuration file path
	def __init__(self, cfg):
		self._batch_size = 10
		pass
		
	# returns batch	corresponding to index
	def get_batch(self, ind):
		pass

	# partition the dataset into batches of size bsize
	def set_batch_size(self, bsize):
		self._batch_size = bsize
		# update other stuff as necessary

		
	def get_batch_size(self):
		return self._batch_size

	
	# returns number of batches per epoch
	def get_num_batches(self):
		pass

	# returns number of examples in epoch
	def get_num_examples(self):
		pass
	
	# shuffles the dataset 
	def shuffle(self):
		pass
