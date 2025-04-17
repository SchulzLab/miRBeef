import re
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

def formatDeepMirTar2(filename, pad_to_len=30):
	"""
	function:
		format the DeepMirTar data: tokenize the sequences into 0,1,2,3,4 and saved the token into a list.
		this function will pad the miRNA sequences with N if they are less than pad_to_len.

	inputs:
		filename - the input file name. The format is as following:
		miRNA Mature_mirna_transcript_reversed gene_Id 3UTR_transcript label
	returns:
		seqs_mirna - A list containing the miRNA sequences
		seqs_targets - A list containing the target sequences
		l - A list containing the labels.
	"""
	seqs = list()
	seqs_mirna = list()
	seqs_targets = list()
	l = list()
	encode = dict(zip('NAUCG', range(5))) # N:0, A:1, U/T:2, C:3, G:4
	with open(filename, 'r') as infl:
		next(infl)
		for line in infl:
			values = line.split("\t")
			if len(values[1]) < pad_to_len:
				values[1] = values[1] + 'N' * (pad_to_len - len(values[1]))
			# Concat padded miRNA seq + unpadded target seq
			# seq2 = values[1] + values[3]
			# seq2 = re.sub('T', 'U', seq2.rstrip('\r\n'))
			#print('seq=' + str(seq2))

			seq_mirna = values[1] ; seq_target = values[3]
			seq_mirna = re.sub('T', 'U', seq_mirna.rstrip('\r\n')) 
			seq_target = re.sub('T', 'U', seq_target.rstrip('\r\n')) 
			# token = [encode[s] for s in seq2.upper()]
			token_mirna = [encode[s] for s in seq_mirna.upper()]
			token_target = [encode[s] for s in seq_target.upper()]
			# name = values[0] + "_" + values[2]
			# seqs.append((token, name))
			# seqs.append((token_mirna, token_target))
			seqs_mirna.append(token_mirna)
			seqs_targets.append(token_target)
			l.append(values[4].rstrip('\r\n'))

	# return seqs, l
	return seqs_mirna, seqs_targets, l

def padding(seqs, maxL=53):
	"""
	function:
		add 0 to the end of the short arrays/list

	inputs:
		seqs - a list of sequences
	outputs:
		a padded array
	"""
	# Originally padded to seq_len that is max among the sequences in the set
	# This can't work if we use CV - right thing to do is to pass a fixed length
	# which depends on the dataset but can be extracted apriori
	# maxL = len(max(seqs, key=len))
	lens = [len(seq) for seq in seqs]
	seqsP = []
	for seq, seq_len in zip(seqs, lens):
		gap = maxL - seq_len
		seq = seq + [0] * gap
		seqsP.append(seq)

	return np.atleast_2d(seqsP)

def load_data(inputf):
    
	seqs_mirna, seqs_targets, label = formatDeepMirTar2(inputf, pad_to_len=26)

	x_mirna = np.array(seqs_mirna)

	# Pad mRNA sequences as well
	x_targets = padding(seqs_targets)
	print(x_mirna.shape)
	print(x_targets.shape)

	# assert (x_mirna.shape[1] == 26)
	# assert (x_targets.shape[1] == 53)

	y = [int(y) for y in label]

	x_mirna = x_mirna.reshape(x_mirna.shape[0], x_mirna.shape[1])
	x_targets = x_targets.reshape(x_targets.shape[0], x_targets.shape[1])
	x_2_left = x_mirna ; x_2_right = x_targets
	y_2 = np.array(y).reshape(len(y), 1)

	return x_2_left, x_2_right, y_2

####################################################################################
## Not used 

# Create a dictionary to map nucleotides to integers
NT_DICT = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

def one_hot_encode(sequences, nt_dict=NT_DICT):
    
    # Create an empty array to store the one-hot encoded sequences
	one_hot = np.zeros((len(sequences), len(sequences[0]), 5))

    # Loop over each sequence
	for i, seq in enumerate(sequences):
        # Loop over each nucleotide in the sequence
		for j, nt in enumerate(seq):
            # Map the nucleotide to an integer using the dictionary
			nt_int = nt_dict[nt]
            # Set the corresponding position in the one-hot array to 1
			one_hot[i, j, nt_int] = 1

	return one_hot

####################################################################################

def generate_mask_indices(unpadded_seq_len, mask_len=2, mask_type='3p'):
	"""Assume we can generate 'mask indices' for both unpadded and padded sequences.
	Padded sequences can contain a stretch of Ns at the bottom. Example:

	unpadded = "AAACGTCG" ; unpadded_seq_len = 8
	padded = "AAACGTNN" ; unpadded_seq_len = 6

	Args:
		unpadded_seq_len (_type_): _description_
		mask_len (int, optional): _description_. Defaults to 2.
		mask_type (str, optional): _description_. Defaults to '3p'.

	Raises:
		ValueError: _description_

	Returns:
		_type_: _description_
	"""

	# Debug
	# print(f"unpadded_seq_len: {unpadded_seq_len}")

	# mask = np.zeros(seq_len)
	# We mask 2nt at the 3'end
	if mask_type == "3p":
		# mask[seq_len-mask_len: ] = 1
		mask_indices = list(range(unpadded_seq_len-1, unpadded_seq_len-1-mask_len, -1))

	# We mask 2nt at the 5' end
	elif mask_type == "5p":
		# mask[0:mask_len] = 1
		mask_indices = list(range(0, mask_len))

	# We mask both
	elif mask_type == "3p5p":
		# mask[0:mask_len] = 1
		# mask[seq_len-mask_len:] = 1
		mask_indices = list(range(0, mask_len))
		mask_indices += list(range(unpadded_seq_len-1, unpadded_seq_len-1-mask_len, -1))
	else:
		raise ValueError(f"Illicit value {mask_type} for mask type argument. Only 3p, 5p, or 3p5p are possible")

	# mask_indices = np.where(mask==1)

	# print(mask_indices)
	return mask_indices

def mask(sequence, len, mask_indices):
	""" Takes a miRNA sequence (ACGT), of a given len, and mask to 'N' all the nucleotides
	corresponding to mask_indices.
	# TODO

	Args:
		sequence (_type_): _description_
		len (_type_): _description_
		mask_indices (_type_): _description_
	"""
	# Change to N all nucleotides at positions indicated by the mask (bit = 1 indicates position to change)

# Consider only canonical base-pairs
pairing_base_dict = {"A": "T", "C": "G", "G": "C", "T": "A"}

def base_complement(sequence, mask_indices):

	# Unpack seq into list of chars or integers
	mut_sequence = [*sequence]
	for i in mask_indices:
		# print(mask_indices)
		# mut_sequence[i] = pairing_base_dict[sequence[i]]
		mut_sequence[i] = pairing_base_int_dict[sequence[i]]
		# print(f"mut_sequence[i]: {mut_sequence[i]}")
	
	# print(mut_sequence)
	try:
		mut_sequence = "".join(mut_sequence)
	except Exception as e:
		# print(e)
		# print("Returning numpy array")
		return mut_sequence

	return mut_sequence

pairing_base_int_dict = {1: 2, 2: 1, 3: 4, 4: 3}

# TODO
def base_complement_one_hot(sequence, mask_indices):
	# UCGUUUUNNN
	print("Not implemented")

def perturb_seq(seq, mask_type="3p", mask_len=2, use_one_hot=False):
	"""Take a sequence of nucleotides and return a perturbed sequence. 
	Assume sequences could be padded with a variable number of Ns at the right end.

	Args:
		seq (_type_): _description_
		seq_len (int, optional): _description_. Defaults to 26.
		mask_type (str, optional): _description_. Defaults to "3p".
		mask_len (int, optional): _description_. Defaults to 2.
		use_one_hot (bool, optional): _description_. Defaults to False.

	Returns:
		_type_: _description_
	"""

	seq_len = len(seq)

	i = seq_len-1
	# Start reading seq from bottom, where the stretch of Ns is
	# while seq[i] == "N":
	# 	i -= 1
	# print(f"seq: {seq}")
	while seq[i] == 0: # 0 = N
		i -= 1
	unpadded_seq_len = i + 1
	mask_indices = generate_mask_indices(unpadded_seq_len, mask_len, mask_type)	

	# Handle the case that the input sequence has already been one-hot encoded
	if use_one_hot:
		mut_seq = base_complement_one_hot(seq, mask_indices)
	else:
		mut_seq = base_complement(seq, mask_indices)

	return mut_seq


def generate_perturb_test_set(X_test_left, mask_type):
	""" Works with X_test dataset obtained after preprocessing data_DeepMirTar_miRAW_noRepeats_3folds.txt.
	Reads in file, apply perturbations to the miRNA sequence (only) and return an updated pertubed test set.

	Args:
		X_test (_type_): _description_
	"""
	# TODO Vectorize instead
	return np.apply_along_axis(perturb_seq, 1, X_test_left, mask_type=mask_type)


def save_kfolds_to_file(path_to_input_dataset, n_folds, output_dir):
	# Take as input either a .txt or .csv file with tab separated columns (like data_DeepMirTar_miRAW_noRepeats_3folds.txt)

	cols = ['mirna_name', 'mirna_seq', 'mrna_ens_id', 'mrna_seq', 'label']
	data = pd.read_csv(path_to_input_dataset, sep="\t", names=cols)

	print(data.head())

	skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)
	target = data.loc[:,'label']

	fold_num = 0
	# Group by target (0/1) to make sure validation and train set have the same ratio of 0:1s
	for train_index, val_index in skf.split(data, target):

		print(len(train_index), len(val_index))
		train = data.loc[train_index,:]
		val = data.loc[val_index,:]
		save_dir = output_dir + "/fold-" + str(fold_num)

		# Create directory, if already existing, leaves the current without overwriting
		os.makedirs(save_dir, exist_ok=True)

		train.to_csv(save_dir + '/train.csv', sep="\t", header=None, index=False)
		val.to_csv(save_dir + '/val.csv', sep="\t", header=None, index=False)
		fold_num += 1