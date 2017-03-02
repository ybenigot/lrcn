from collections import OrderedDict
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import math as m
import os
import random

# a simpler version of captioner.py from caffe coco_caption 

class SimpleCaptioner():

	def __init__(self, net, vocab_path):

		# Setup sentence prediction net.
		self.lstm_net = net
		self.vocab = ['<EOS>']
		self.start=True
		try:
			# Compute vocabulary
			with open(vocab_path, 'r') as vocab_file:
				self.vocab += [word.strip() for word in vocab_file.readlines()]
			net_vocab_size = self.lstm_net.blobs['predict'].data.shape[2]
			if len(self.vocab) != net_vocab_size:
				print 'Invalid vocab file: contains %d words; '
				print 'net expects vocab with %d words' % (len(self.vocab), net_vocab_size)
		except:
			print '**** cannot read vocabulary'

	def predict_single_word(self, previous_word, image_rep):
		'''predict one word at a time, using the LSTM layers
			descriptor an image vector representation as an input to the LSTMs
		'''
		net = self.lstm_net
		cont = 0 if previous_word == 0 else 1
		cont_input = np.array([cont])
		word_input = np.array([previous_word])
		net.blobs['input_sentence'].data[0, :]=word_input
		net.blobs['cont_sentence'].data[0, :]=cont_input
		net.blobs['score'].data[0,:]= image_rep 
		net.forward(start='embedding',end='softmax')
		output_preds = net.blobs['softmax'].data[0, 0, :]
		return output_preds

	def sample_caption(self, image_rep, max_length=50):
		sentence = []
		if self.start:
			self.first_rep = np.copy(image_rep)
			self.start = False
		else:
			sq = (self.first_rep - image_rep) * (self.first_rep - image_rep)
			dist=m.sqrt(np.sum(sq) / np.sum (image_rep * image_rep))
			print 'distance %f ' % (dist,)
		print image_rep[500:510]
		while len(sentence) < max_length and (not sentence or sentence[-1] != 0):
			previous_word = sentence[-1] if sentence else 0
			output_preds = self.predict_single_word(previous_word,image_rep)
			word = np.argmax(output_preds)
			sentence.append(word)
		return sentence

	def get_word(self,index):
		if index < 0 :
			return 'UNK'
		else:
			return self.vocab[index]

	def get_sentence(self,word_list):
		str=""
		for word in word_list:
			str+=self.get_word(word)
			str+=" "
		return str		
