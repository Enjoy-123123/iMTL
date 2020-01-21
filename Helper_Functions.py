import numpy as np
import pandas as pd
import progressbar
import itertools
import math
import collections
from six.moves import cPickle as pickle # for pickleperformance
import random

# ==============================================
# 1. Data Preprocessing
# ==============================================

def check_is_consecutive(check_list, start_index):
	"""
	Purpose:
	check if an integer(ID) list is consecutive

	Parameters:
	check_list (ndarray<int>): 1d array to be checked
	start_index (int): the first index of the consecutive id

	Returns:
	nothing if the list passes consecutive-ID check
	throw an AssertionError if check fails          

	"""
	
	assert check_list.max() == len(np.unique(check_list)) + start_index - 1, 'ID is not consecutive'

# end def


def _remove_consecutive_visit(visit_record, bar):
	"""
	Purpose:
	remove consecutive visits (to the same POI) in a visit sequence

	Parameters:
	visit_record (df) : record of a visit sequence
	bar: (ProgressBar) : used for progress display

	Returns:
	clean_sequence (list<int(index)>) : processed sequence without consecutive visits

	"""
	
	clean_sequence = []
	
	for index,visit in visit_record.iterrows():
		bar.update(index)
		if index==visit_record.index[0]: # skip first row 
			continue 
		elif visit['POI_id'] != visit_record['POI_id'][index-1]: # only accept non-repeated visit
			clean_sequence.append(index-1)
			
	return clean_sequence

# end def


def generate_sequence(input_data, min_seq_len, min_seq_num):
	"""
	Purpose:
	generate visit sequences for each user with the following filtering rules:
	1. visits within 1 day for a user is considered a single visit sequence
	2. consecutive visits to the same POI should be removed
	3. length of visit sequence > 'min_seq_len' is considered valid sequence
	4. number of sequences > 'min_seq_num' is considered a valid user
	5. only valid sequences and from valid users will become training/test sequences

	Parameters:
	input_data (dataframe) : initial data loaded from 'data_SIN.csv'

	Returns:
	total_sequences_dict (dictionary {
						(int):User_id, 
						(ndarray[seq_num, visit_index]): filtered visit index of each valid sequence
					}) : records each valid users' valid visiting sequences
	max_seq_len (int) : length of the longest sequence (for padding)
	valid_visits (list<int>) : 1d list of all valid visit index (to filter valid visit information)
	user_reIndex_mapping (ndarray<int>[old_user_id]) : 1d array mapping old user index (array value) to new user index (array index)
	"""
	
	# setup progress bar
	bar = progressbar.ProgressBar(maxval=input_data.index[-1], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	
	# convert time string to Datetime object for comparison
	input_data['Local_sg_time'] = pd.to_datetime(input_data['Local_sg_time'])
	
	total_sequences_dict = {} # record all sequences
	
	max_seq_len = 0 # record maximum sequence length

	valid_visits = [] # record all valid visit index
	
	bar.start()
	
	# find visit sequences for each user
	for user in input_data['User_id'].unique():
		user_visits = input_data[input_data['User_id'] == user]
		user_sequences = [] # record a users' sequences
		
		# find each visit sequence for each user:
		unique_date_group = user_visits.groupby([user_visits['Local_sg_time'].dt.date]) #group visits by date
		
		for date in unique_date_group.groups:
			single_date_visit = unique_date_group.get_group(date)
			single_sequence = _remove_consecutive_visit(single_date_visit, bar) # reovme consecutive visits in each sequence
			
			if len(single_sequence) >= min_seq_len: # add previous sequence (if valid) to user_sequences
				user_sequences.append(single_sequence)
				if len(single_sequence) > max_seq_len: # update max sequence length
					max_seq_len = len(single_sequence) 
			
		if len(user_sequences) >= min_seq_num: # add valid user to final output
			total_sequences_dict[user]=np.array(user_sequences)
			valid_visits = valid_visits + list(itertools.chain.from_iterable(user_sequences))

	bar.finish()

	user_reIndex_mapping = np.array(list(total_sequences_dict.keys()))
	
	return total_sequences_dict, max_seq_len, valid_visits, user_reIndex_mapping

# end def


def aug_sequence(input_sequence_dict, min_len):
	"""
	Purpose:
	augment each sequence to increase sample size. 
	For example, [0,1,2,3,4] will be augmented to: [0,1,2],[0,1,2,3],[0,1,2,3,4]

	Parameters:
	input_sequence_dict (dictionary {
							(int):User_id, 
							(ndarray[seq_num, seq_len]): filtered visit index of each valid sequence
						}) : records each valid users' valid visiting sequences
	min_len (int) : minimum length of the augmented sequences 

	Returns:
	augmented_sequence_dict (dictionary {
								(int):User_id, 
								(ndarray[augmented_seq_num, seq_len]): filtered visit index of each valid sequence
							}) : sequence dictionary with augmented sequences

	ground_truth_dict (dictionary {
							(int):User_id, 
							(ndarray[augmented_seq_num, seq_len]): filtered visit index of ground_truth of each valid sequence
						}) : sequence dictionary with ground truth of each augmented sequences
	"""
	
	augmented_sequence_dict, ground_truth_dict = {}, {}
	
	for user in input_sequence_dict.keys():
		
		user_sequences, ground_truth_sequence = [], []
		
		#data augmentation: add more sequence from existing sequence
		for seq in input_sequence_dict[user]:
			if len(seq)>min_len:
				for i in range(len(seq)-min_len+1): 
					user_sequences.append(seq[0:i+min_len])
					ground_truth_sequence.append(seq[i+min_len-1:])
			else: 
				user_sequences.append(seq)
				ground_truth_sequence.append([seq[-1]])
		
		augmented_sequence_dict[user] = np.array(user_sequences)
		ground_truth_dict[user] = np.array(ground_truth_sequence)
		
	return augmented_sequence_dict, ground_truth_dict

# end def


def pad_sequence(input_sequence_dict, max_seq_len):
	"""
	Purpose:
	pad sequences with -1 to form uniform sequence length (for model inputs)

	Parameters:
	input_sequence_dict (dictionary {
							(int):User_id, 
							(ndarray[seq_num, seq_len]): filtered visit index of each valid sequence
						}) : records each valid users' valid visiting sequences
	max_seq_len (int): length of the longest sequence
	
	Returns:
	padded_sequence_dict (dictionary {
							(int):User_id, 
							(ndarray[seq_num, longest_seq_len]): filtered visit index of each valid sequence
						 }) : padded sequence dictionary recording each valid users' valid visiting sequences
	"""
	
	padded_sequence_dict = {}
	
	# perform padding on each sequence
	for user in input_sequence_dict.keys():
		
		user_sequences = []
		
		for seq in input_sequence_dict[user]:
			seq = np.pad(seq, (0,max_seq_len - len(seq)), 'constant', constant_values=-1,)
			user_sequences.append(seq)
			
		padded_sequence_dict[user] = np.array(user_sequences)
	
	return padded_sequence_dict

# end def


def peep_dictionary(input_dict):
	"""
	Purpose:
	print first entry for the input dictionary (for checking)

	Parameters:
	input_dict (dictionary): dictionary to be checked
	
	Returns:
	null
	"""
	
	print(next(iter(input_dict)), ' :\n', input_dict[next(iter(input_dict))])
	print('dictionary size: ', len(input_dict.keys()))

# end def

	
def _flatten_3d_list(input_list):
	"""
	Purpose:
	flatten a 3d list into 1d list

	Parameters:
	input_list (ndarray<int>[d1,d2,d3]): a ununiform 3d list to be flattened. e.g., [users,seqs,visit]
	
	Returns:
	flattened_list (ndarray<int>[d1]): the flattend 1d list
	"""
	
	twoD_lists = input_list.flatten()
	
	return np.hstack([np.hstack(twoD_list) for twoD_list in twoD_lists])

# end def


def _old_id_to_new(mapping, old_id):
	"""
	Purpose:
	given an old id and a mapping, return the new id

	Parameters:
	mapping (ndarray<int>) : a 1d array mapping old id (array value) to new id (array index) 
	old_id (int) : old id to be converted
	
	Returns:
	new_id (int): new id mapped from the old id
	"""
  
	return np.where(mapping == old_id)[0].flat[0]

# end def


def _new_id_to_old(mapping, new_id):
	"""
	Purpose:
	given an old id and a mapping, return the new id

	Parameters:
	mapping (ndarray<int>) : a 1d array mapping old id (array value) to new id (array index) 
	new_id (int) : new id to be converted to old id
	
	Returns:
	old_id (int): old id mapped from the new id
	"""
  
	return mapping[new_id]

# end def


def _reIndex_3d_list(input_list):
	"""
	Purpose:
	reIndex an id-list to form consecutive id started from 0
	generate a map to match new and old ids
	check correctness of the reindex before output

	Parameters:
	input_list (ndarray<int>[user,seq,visit]) : a ununiform 3d array to be reindexed
	
	Returns:
	reIndexed_list (ndarray<int>[user,seq,visit]): a ununiform reindexed 3d array
	index_map (ndarray<int>[old_index]) : a map of new and old id, where new id is list index and 
	"""
	
	# flatten list to 1d
	flat_list = _flatten_3d_list(input_list)
	
	# generate mapping from old to new id
	index_map = np.unique(flat_list)

	if index_map[0] == -1: # remove padding from index_map
		index_map = np.delete(index_map, 0)
	
	# update list with new id
	reIndexed_list = [] # reIndexed list for all users
	
	for user in input_list:
		
		reIndexed_user_list = [] # reIndexed list for each user
		
		for seq in user:
			reIndexed_user_list.append([_old_id_to_new(index_map, poi) if poi != -1 else -1 for poi in seq])
			
		reIndexed_list.append(reIndexed_user_list)
		
	reIndexed_list = np.array(reIndexed_list)
		
	# varify reindexing correctness
	check_list = _flatten_3d_list(reIndexed_list)

	if -1 in check_list: # remove paddings to perform check
		check_list = check_list[ check_list >= 0 ]
	
	check_is_consecutive(check_list, 0) 
	
	# output 
	return reIndexed_list, index_map

# end def
  
  
def generate_POI_sequences(input_data, visit_sequence_dict):
	"""
	Purpose:
	reindex POIs to form  consecutive ids started from 0
	record old and new id mapping for back tracing
	generate POI id (with new id) sequences for each valid user 

	Parameters:
	input_data (dataframe) : initial data loaded from 'data_SIN.csv'
	visit_sequence_dict (dictionary {
							(int):User_id, 
							(ndarray<int>[seq_num, seq_len]): visit index of each valid sequence
						  }) : records each valid users' valid visiting sequences
	
	Returns:
	reIndexed_POI_sequences (ndarray<int>[user,seq,visit]): array recording POI id (reindexed) of each poi in visit sequence
	POI_reIndex_mapping (ndarray<int>[old_poi_index]) : 1d array mapping old POI index (array value) to new POI index (array index)
	"""
	
	POI_sequences = []
	
	for user in visit_sequence_dict:
		
		user_POI_sequences = []
		
		for seq in visit_sequence_dict[user]:
			
			POI_sequence = []
			
			for visit in seq:
				if visit != -1:
					POI_sequence.append(input_data['POI_id'][visit])
				else: # ignore padding
					POI_sequence.append(-1)
			
			user_POI_sequences.append(POI_sequence)
		
		POI_sequences.append(user_POI_sequences)
	
	# reindex POI sequences

	reIndexed_POI_sequences, POI_reIndex_mapping = _reIndex_3d_list(np.array(POI_sequences))
	
	return reIndexed_POI_sequences, POI_reIndex_mapping
  
# end def
  
  
def _haversine(pos1, pos2):
	"""
	Purpose:
	calculate haversine distance between to points
	haversine distance is the straight line distance between two points on a sphere:
	a = sin²(Δlat/2) + cos lat1 ⋅ cos lat2 ⋅ sin²(Δlon/2)
	c = 2 ⋅ atan2( √a, √(1−a) )
	d = R ⋅ c, R is earth's radius = 6371

	Parameters:
	pos1 (tuple<float,float>) : latitude and longitude (signed degree) of current poi
	pos2 (tuple<float,float>) : latitude and longitude (signed degree) of previous poi

	Returns:
	h_dist (float): 

	"""
	
	lat1, lon1 = pos1
	lat2, lon2 = pos2
	
	dlat = lat2 - lat1
	dlon = lon2 - lon1
	
	a = math.sin(math.radians(dlat / 2)) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(math.radians(dlon / 2)) ** 2
	c = 2 * math.asin(math.sqrt(a))
	r = 6371 
	h_dist = c * r
	
	return h_dist
  
#end def


def generate_dist_sequences(input_data, visit_sequence_dict):
	"""
	Purpose:
	generate dist sequences for each valid user
	dist sequence records distance (ceiling to km) of each poi with its previous poi in visit sequence
	start point of sequence has distance 0
	record maximum distance to decide distance embedding dimension

	Parameters:
	input_data (dataframe) : initial data loaded from 'data_SIN.csv'
	visit_sequence_dict (dictionary {
							(int):User_id, 
							(ndarray[seq_num, seq_len]): visit index of each valid sequence
						  }) : records each valid users' valid visiting sequences
	
	Returns:
	dist_sequences (ndarray<int>[user,seq,visit]) : array recording distance (ceiling to km) of each poi with its previous poi in visit sequence
	max_dist (int) : maximum distance (to decide embedding dimension)
	"""
	
	dist_sequences = []
	max_dist = 0
	
	for user in visit_sequence_dict:
		
		user_dist_sequences = []
		
		for seq in visit_sequence_dict[user]:
			
			dist_sequence = []
			
			for pos, visit in enumerate(seq):
				if pos == 0: # start point of sequence has distance 0
					dist_sequence.append(0)
					
				elif visit != -1:
					lat1 = input_data['Latitude'][visit]
					lon1 = input_data['Longitude'][visit]
					lat2 = input_data['Latitude'][seq[pos-1]]
					lon2 = input_data['Longitude'][seq[pos-1]]
					
					dist = _haversine((lat1,lon1), (lat2,lon2))
					dist_sequence.append(math.ceil(dist))
					max_dist = max(max_dist, math.ceil(dist))

				else: # ignore padding
					dist_sequence.append(-1)
			
			user_dist_sequences.append(dist_sequence)
		
		dist_sequences.append(user_dist_sequences)
	
	return np.array(dist_sequences), max_dist
  
# end def


def generate_type_sequence(input_data, visit_sequence_dict):
	"""
	Purpose:
	generate type sequences for each valid user
	1 for Collective POI, 0 for Individual POI

	Parameters:
	input_data (dataframe) : initial data loaded from 'data_SIN.csv'
	visit_sequence_dict (dictionary {
							(int):User_id, 
							(ndarray[seq_num, seq_len]): visit index of each valid sequence
						  }) : records each valid users' valid visiting sequences
	
	Returns:
	type_sequences (ndarray<int>[user,seq,visit]): array recording type (0,1) of each poi in visit sequence
	"""
	
	type_sequences = []
	
	for user in visit_sequence_dict:
		
		user_type_sequences = []
		
		for seq in visit_sequence_dict[user]:
			
			type_sequence = []
			
			for visit in seq:
				if visit != -1:
					type_sequence.append(int(input_data['POI_Type'][visit]=='Combined'))
				else: # ignore padding
					type_sequence.append(-1)
			
			user_type_sequences.append(type_sequence)
		
		type_sequences.append(user_type_sequences)
	
	return np.array(type_sequences)
  
# end def


def generate_time_sequences(input_data, visit_sequence_dict):
	"""
	Purpose:
	generate time sequences for each valid user
	time sequence is records visit time (discretised into hours) for each visit in a sequence
	
	Parameters:
	input_data (dataframe) : initial data loaded from 'data_SIN.csv'
	visit_sequence_dict (dictionary {
							(int):User_id, 
							(ndarray[seq_num, seq_len]): visit index of each valid sequence
						  }) : records each valid users' valid visiting sequences
	
	Returns:
	time_sequences (ndarray<int>[user,seq,visit]): array recording visit time (0-23 in hour) of each poi in visit sequence
	"""
	
	# convert time string to Datetime object to get visit hour
	input_data['Local_sg_time'] = pd.to_datetime(input_data['Local_sg_time'])
	
	time_sequences = []
	
	for user in visit_sequence_dict:
		
		user_time_sequences = []
		
		for seq in visit_sequence_dict[user]:
			
			time_sequence = []
			
			for visit in seq:
				if visit != -1:
					time_sequence.append(input_data['Local_sg_time'][visit].hour)
				else: # ignore padding
					time_sequence.append(-1)
			
			user_time_sequences.append(time_sequence)
		
		time_sequences.append(user_time_sequences)
	
	return np.array(time_sequences)
  
# end def


def generate_cat_sequences(input_data, visit_sequence_dict):
	"""
	Purpose:
	generate categoory sequences for each valid user
	category sequence records categories of pois in a visit sequence

	Parameters:
	input_data (dataframe) : initial data loaded from 'data_SIN.csv'
	visit_sequence_dict (dictionary {
							(int):User_id, 
							(ndarray[seq_num, seq_len]): visit index of each valid sequence
						  }) : records each valid users' valid visiting sequences
	
	Returns:
	reIndexed_cat_sequences (ndarray<int>[user,seq,visit]): array recording visit category (reindexed) of each poi in visit sequence
	cat_reIndex_mapping (ndarray<int>[old_cat_index]) : 1d array mapping old category index (array value) to new category index (array index)
	"""
	
	cat_sequences = []
	
	for user in visit_sequence_dict:
		
		user_cat_sequences = []
		
		for seq in visit_sequence_dict[user]:
			
			cat_sequence = []
			
			for visit in seq:
				if visit != -1:
					cat_sequence.append(input_data['L2_id'][visit])
				else: # ignore padding
					cat_sequence.append(-1)
			
			user_cat_sequences.append(cat_sequence)
		
		cat_sequences.append(user_cat_sequences)

	reIndexed_cat_sequences, cat_reIndex_mapping = _reIndex_3d_list(np.array(cat_sequences))
	
	return reIndexed_cat_sequences, cat_reIndex_mapping
  
# end def


def generate_ground_truth_sequences(input_data, ground_truth_dict, POI_reindex_mapping):
	"""
	Purpose:
	
	
	Parameters:
	
	
	Returns:
	
	"""
	
	ground_truth_sequences = []
	
	for user in ground_truth_dict:
		
		user_ground_truth_sequence = []
		
		for seq in ground_truth_dict[user]:
			
			ground_truth_sequence = []
			
			for visit in seq:
				if visit != -1:
					ground_truth_sequence.append(_old_id_to_new(POI_reindex_mapping, input_data['POI_id'][visit]))
				else: # ignore padding
					ground_truth_sequence.append(-1)
			
			user_ground_truth_sequence.append(ground_truth_sequence)
		
		ground_truth_sequences.append(user_ground_truth_sequence)
		
	return ground_truth_sequences
		
# end def

def generate_specific_poi_sequences(input_data, ground_truth_dict):

	specific_poi_sequences = []
	
	for user in ground_truth_dict:
		
		user_ground_truth_sequence = []
		
		for seq in ground_truth_dict[user]:
			
			ground_truth_sequence = []
			
			for visit in seq:
				if visit != -1:
					ground_truth_sequence.append(input_data['Location_id'][visit])
				else: # ignore padding
					ground_truth_sequence.append(-1)
			
			user_ground_truth_sequence.append(ground_truth_sequence)
		
		specific_poi_sequences.append(user_ground_truth_sequence)
		
	return specific_poi_sequences

# end def


def generate_cat_distrib(input_data, valid_visits, POI_reIndex_mapping, cat_reIndex_mapping):
	"""
	Purpose:
	generate category (L2) distribution for each collective POI
	
	Parameters:
	input_data (dataframe) : initial data loaded from 'data_SIN.csv'
	valid_visits (list<int>) : 1d list records index of all valid visit
	POI_reIndex_mapping (ndarray<int>[old_poi_index]) : 1d array mapping old POI index (array value) to new POI index (array index)
	cat_reIndex_mapping (ndarray<int>[old_cat_index]) : 1d array mapping old category index (array value) to new category index (array index)
	
	Returns:
	all_poi_cat_distrib (dictionary {
							(int): new POI id,
							dictionary {
								(int) : new category id,
								(int) : count of store number under the category
							}
						}): records the distribution of categoryies under each collective POI
	"""

	all_poi_cat_distrib = {}

	# get only the valid visits

	valid_data = input_data[input_data.index.isin(valid_visits)]

	# find all collective POIs

	collective_POI_visits = valid_data[valid_data['POI_Type'] == 'Combined']

	# for each collective POI, count No. POIs under each category
	for collective_POI in collective_POI_visits['POI_id'].unique():

		collective_POI_visit = collective_POI_visits[collective_POI_visits['POI_id'] == collective_POI]

		# replace old category id with new id
		collective_POI_visit['L2_id'] = collective_POI_visit['L2_id'].apply(lambda x: _old_id_to_new(cat_reIndex_mapping, x))

		poi_cat_distrib = collections.Counter(collective_POI_visit['L2_id'])

		all_poi_cat_distrib[_old_id_to_new(POI_reIndex_mapping, collective_POI)] = poi_cat_distrib

	return all_poi_cat_distrib

# end def

def generate_POI_dist_mat(input_data, POI_reIndex_mapping):
	"""
	Purpose:
	generate a matrix storing distance between each POI
	
	Parameters:
	input_data (dataframe) : initial data loaded from 'data_SIN.csv'
	valid_visits (list<int>) : 1d list of all valid visit index (to filter valid visit information)
	POI_reIndex_mapping (ndarray<int>[old_poi_index]) : 1d array mapping old POI index (array value) to new POI index (array index)

	Returns:
	POI_dist_mat (ndarray<float>[poi,poi]) : 2d array storing distance between each pois
	"""

	POI_num = len(POI_reIndex_mapping)

	POI_dist_mat = np.zeros([POI_num, POI_num])

	# setup progress bar
	bar = progressbar.ProgressBar(maxval = POI_num, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

	bar.start()

	for i in range(1,POI_num):

		bar.update(i)

		for j in range(i): # compute distance between i and j

			lat1 = input_data[input_data['POI_id'] == _new_id_to_old(POI_reIndex_mapping,i)].iloc[0]['Latitude']
			lon1 = input_data[input_data['POI_id'] == _new_id_to_old(POI_reIndex_mapping,i)].iloc[0]['Longitude']
			lat2 = input_data[input_data['POI_id'] == _new_id_to_old(POI_reIndex_mapping,j)].iloc[0]['Latitude']
			lon2 = input_data[input_data['POI_id'] == _new_id_to_old(POI_reIndex_mapping,j)].iloc[0]['Longitude']

			POI_dist_mat[i][j] = _haversine((lat1,lon1),(lat2,lon2))

	bar.finish()

	return POI_dist_mat

# end def


def generate_neg_sequences(POI_sequences, POI_dist_mat, neg_sample_num, input_data, POI_reIndex_mapping, cat_reIndex_mapping):
	"""
	Purpose:
	form negative samples for each visit sequence. 
	negative sample is a POI id which satisfies the following criteria:
	1. The POI does not appear in the true sequence 
	2. The distance between:
		a) negative POI and true destination  
		b) true second last POI and true destination
	should be as close as possible
	
	Parameters:
	POI_sequences (ndarray<int>[user,seq,visit]) : array recording POI id (reindexed) of each poi in visit sequence
	POI_dist_mat (ndarray<float>[poi,poi]) : 2d array storing distance between each pois
	neg_sample_num (int) : number of negative samples per sequence
	input_data (dataframe) : initial data loaded from 'data_SIN.csv'
	POI_reIndex_mapping (ndarray<int>[old_poi_index]) : 1d array mapping old POI index (array value) to new POI index (array index)
	cat_reIndex_mapping (ndarray<int>[old_cat_index]) : 1d array mapping old category index (array value) to new category index (array index)
		
	Returns:
	total_neg_sequences (ndarray<int>[user,seq,poi,poi_info(id, cat, type)]) : 4d array storing negative samples for each sequence
	"""

	total_neg_sequences = []

	POI_num = POI_dist_mat.shape[0] # number of all POIs

	for user in POI_sequences:

		user_neg_sequences = []

		for seq in user:
			# get true distance to the destination
			no_pad_seq = [x for x in seq if x != -1] # in case of padding

			dest = no_pad_seq[-1]
			sec_last = no_pad_seq[-2]
			true_dist = POI_dist_mat[dest][sec_last]

			# get candidates' distance to destination
			neg_cand = list(set(np.arange(POI_num)) - set(seq)) # get candidates not in true sequence
			candidate_dist = [(POI_dist_mat[dest][cand] - true_dist) if cand in neg_cand else 10000 for cand in np.arange(POI_num)] # calculate all negative candidates' distance to destination. If not candidate, assign a arbitrarily large value

			# select top k qualified samples
			sorted_index = np.argsort(candidate_dist) # sort candidates according to distance diff
			neg_poi_sequence = sorted_index[:neg_sample_num] # get closest candidates
			neg_sequence = [] # store information for each negative poi

			for poi in neg_poi_sequence:
				# get category and type of each negative POI
				poi_entry = input_data[input_data['POI_id'] == _new_id_to_old(POI_reIndex_mapping, poi)].iloc[0]
				poi_type = int(poi_entry['POI_Type'] == 'Combined')
				if poi_type: 
					poi_cat = -1
				else:
					poi_cat = _old_id_to_new(cat_reIndex_mapping, poi_entry['L2_id'])

				neg_sequence.append([poi, poi_cat, poi_type])

			user_neg_sequences.append(neg_sequence)

		total_neg_sequences.append(user_neg_sequences)

	return total_neg_sequences

# end def


def form_sample_sets(POI_sequences, dist_sequences, time_sequences, type_sequences, cat_sequences, ground_truth_sequences, specific_poi_sequences, neg_sequences):
	"""
	Purpose:
	form sample set for each valid user
	a sample set consists of 
	
	Parameters:
	POI_sequences (ndarray<int>[user,seq,visit]): array recording POI id (reindexed) of each poi in visit sequence
	dist_sequences (ndarray<int>[user,seq,visit]) : array recording distance (ceiling to km) of each poi with its previous poi in visit sequence
	type_sequences (ndarray<int>[user,seq,visit]): array recording type (0,1) of each poi in visit sequence
	time_sequences (ndarray<int>[user,seq,visit]): array recording visit time (0-23 in hour) of each poi in visit sequence
	cat_sequences (ndarray<int>[user,seq,visit]): array recording visit category (reindexed) of each poi in visit sequence
	neg_sequences (ndarray<int>[user,seq,visit]) : array storing negative samples for each sequence
	
	Returns:
	all_poi_cat_distrib (dictionary {
							(int): new user Id,
							(list<tuple(
								(ndarray<int>[seq,visit]) : POI sequence
								(ndarray<int>[seq,visit]) : distance sequence
								(ndarray<int>[seq,visit]) : time sequence
								(ndarray<int>[seq,visit]) : type sequence
								(ndarray<int>[seq,visit]) : cat sequence
							)>)
						}): collection of a sample for model training input
	"""

	sample_set = {} 

	user_count = 0
	sample_count = 0

	for user_pos, user in enumerate(POI_sequences):

		user_sample = []

		for seq_pos, seq in enumerate(user):
			
			user_sample.append((POI_sequences[user_pos][seq_pos], 
								dist_sequences[user_pos][seq_pos], 
								time_sequences[user_pos][seq_pos],
								type_sequences[user_pos][seq_pos],
								cat_sequences[user_pos][seq_pos],
								ground_truth_sequences[user_pos][seq_pos],
								specific_poi_sequences[user_pos][seq_pos],
								neg_sequences[user_pos][seq_pos]
							))
			sample_count += 1

		sample_set[user_count] = user_sample
		user_count += 1

	print('Total user: %d -- Total sample: %d' %(user_count, sample_count))

	return sample_set

# end def


def save_dict(dic, path):
	"""
	Purpose:
	save a dictionary to a static file
	
	Parameters:
	dic (dictionary) : dictionary to be saved
	path (string) : destination of the path
	
	Returns:
	null
	"""

	with open(path, 'wb') as f:
		pickle.dump(dic, f)

# end def

# ==============================================
# 2. Model construction
# ==============================================

def load_dict(path):
	"""
	Purpose:
	load a dictionary from a static file
	
	Parameters:
	path (string) : path of the static file
	
	Returns:
	dic (dictionary) : loaded dictionary
	"""

	with open(path, 'rb') as f:
		dic = pickle.load(f)
	return dic

# end def


def shuffle(input):
	"""
	Purpose:
	shuffle an input array/list/tuple
	
	Parameters:
	shuffled (array/list/tuple) : list to be shuffled
	
	Returns:
	null
	"""

	random.seed(2019)
	random.shuffle(input)
	return input

# end def


def _shuffle(input):
	"""
	Purpose:
	shuffle an input array/list/tuple
	
	Parameters:
	shuffled (array/list/tuple) : list to be shuffled
	
	Returns:
	null
	"""

	random.seed(2019)
	random.shuffle(input)
	return input

# end def

# class Batch_Generator(self):
	
#     def __init__(self, samples, dist_mat, cat_distrib, is_training = True):
	
#         self.i = 0  # record position 
		
#         X_train, X_test, y_train, y_test, neg_samples = _generate_train_test(samples)
		
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train
#         self.y_test = y_test
#         self.neg_samples = neg_samples
		
#     def next_batch(self, batch_size):

#     	x_samples = self.X_train
#     	y_samples = self.y_train
#     	if not is_training:
#     		x_samples = self.X_test
#     		y_samples = self.y_test
		
#         feed_dict = {}
		
#         feed_dict['x_poi'] = x_samples[self.i : self.i + batch_size]
#         feed_dict['x_dist'] = x_samples[self.i : self.i + batch_size]
#         feed_dict['x_time'] = x_samples[self.i : self.i + batch_size]
#         feed_dict['x_type'] = x_samples[self.i : self.i + batch_size]
#         feed_dict['x_cat'] = _generate_x_cat_vec(x_samples)
		
#         feed_dict['y_poi'] = y_samples[self.i : self.i + batch_size]
#         feed_dict['y_dist'] = y_samples[self.i : self.i + batch_size]
#         feed_dict['y_time'] = y_samples[self.i : self.i + batch_size]
#         feed_dict['y_type'] = y_samples[self.i : self.i + batch_size]
#         feed_dict['y_cat'] = _generate_y_cat_vec(y_samples)
		
#         feed_dict['neg_poi'] = 
#         feed_dict['neg_dist'] = 
#         feed_dict['neg_time'] = _generate_cat_vec(x_samples)
#         feed_dict['neg_type'] = 
#         feed_dict['neg_cat'] = 

#         self.i += batch_size
		
#         return feed_dict
		
#     def has_next(self):
		
#         return i > 

#     def train_test_split(samples, train_portion):
#     	"""
#     	Purpose:
#     	split train test set after shuffling 
		
#     	Parameters:
#     	shuffled (array/list/tuple) : list to be shuffled
		
#     	Returns:
#     	train_samples (tuple) : training samples
#     	test_samples (tuple) : testing samples
#     	"""

#     	# shuffle samples
#     	samples = _shuffle(samples)

#     	# split train test
#     	N = len(samples)
#     	last_train = int(train_portion * N)

#     	train_samples = user0_sample[ : last_train]
#     	test_samples = user0_sample[last_train : ]

#     	# get 
#     	sample_len = len(user0_sample_train)
#     	if -1 in user0_sample_train:
#     		sample_len = sample[0].index(-1)
#     	poi_y = sample[0][sample_len]
#     	poi_x = sample[0][:sample_len] + sample[0][sample_len + 1:]
#     	dist_y = sample[1][sample_len]
#     	dist_x = sample[1][:sample_len] + sample[0][sample_len + 1:]
#     	time_y = sample[2][sample_len]
#     	time_x = sample[2][:sample_len] + sample[0][sample_len + 1:]
#     	type_y = sample[1][sample_len]
#     	type_x = sample[1][:sample_len] + sample[0][sample_len + 1:]

#     	return X_train, X_test, y_train, y_test, neg_samples

#     # end def

	
#     def _generate_train_test(self, samples):
		
#         samples_train, samples_test = Helper.train_test_split(samples, train_portion)
		
#         return X_train, X_test, y_train, y_test, neg_samples
		
#     def _generate_cat_vec(self, ):
		
		