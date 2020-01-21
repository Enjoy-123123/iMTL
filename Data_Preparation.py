# dependencies
import numpy as np
import pandas as pd
import os
# import argparse

# import helper functions

import Helper_Functions as Helper


# ### Adjustable parameters:
# 
# 1. **small_sample** *(boolean)*: Whether to use a small sample (1000 visits) for testing
# 2. **augment_sample** *(boolean)*: Whether to perform sample augmentation
# 3. **pad_data** *(boolean)*: Whether to perform padding on data sequence
# 
# 4. **min_seq_len** *(int)*: Minimum No. POIs for a valid sequence
# 5. **min_seq_num** *(int)*: Minimun No. valid sequences for a valid user
# 6. **neg_sample_num** *(int)*: Number of negative samples for each POI
# setup parameters (for terminal execution)

parser = argparse.ArgumentParser(description='Settings of data processing')

parser.add_argument('--small_sample',type=int,default=False,help='Whether to use a small sample (1000 visits) for testing')
parser.add_argument('--augment_sample',type=bool,default=False,help='Whether to perform sample augmentation')
parser.add_argument('--pad_data',type=bool,default=False,help='Whether to perform padding on data sequence')
parser.add_argument('--min_seq_len',type=int,default=2,help='Minimum No. POIs for a valid sequence')
parser.add_argument('--min_seq_num',type=int,default=2,help='Minimun No. valid sequences for a valid user')
parser.add_argument('--neg_sample_num',type=int,default=5,help='Number of negative samples for each POI')

args = parser.parse_args()

# setup parameters (for ipython execution)

small_sample = False
augment_sample = True
pad_data = False

min_seq_len = 2
min_seq_num = 2
neg_sample_num = 5


# ## 1.Import data


if small_sample:  
    data = pd.read_csv('./data_CHA.csv')[:20000] 
else: 
    data = pd.read_csv('./data_CHA.csv')


# ## 2. Generate Visit Sequence 
# Generate valid index sequences for each valid user
# check consecutiveness of User_id, Location_id, POI_id, L1_id, L2_id (only when full run)

if not(small_sample): 
    
    check_columns = ['User_id','Location_id','POI_id','L1_id','L2_id']

    for col in check_columns:
        Helper.check_is_consecutive(np.array(data[col]), 1)

# form visit sequences 

visit_sequences, max_seq_len, valid_visits, user_reIndex_mapping = Helper.generate_sequence(data, min_seq_len, min_seq_num)

assert bool(visit_sequences), 'no qualified sequence after filtering!' # check if output sequence is empty



Helper.peep_dictionary(visit_sequences)


# augment sequences (optional)

if augment_sample:
#     visit_sequences = Helper.aug_sequence(visit_sequences, min_len=3)
    visit_sequences, ground_truth_dict = Helper.aug_sequence(visit_sequences, min_len=3)


Helper.peep_dictionary(visit_sequences)



Helper.peep_dictionary(ground_truth_dict)



# pad sequences (optional)

if pad_data:
    
    visit_sequences = Helper.pad_sequence(visit_sequences, max_seq_len)



Helper.peep_dictionary(visit_sequences)


# ## 3. Prepare Input Sequences
# Five input sequences paralleled with the Visit Sequence are prepared:
# 1. POI sequence
# 2. Distance sequence
# 3. Time sequence
# 4. Type sequence
# 5. Category sequence


# generate POI sequence

POI_sequences, POI_reIndex_mapping = Helper.generate_POI_sequences(data, visit_sequences)



# generate distance sequence

dist_sequences, max_dist = Helper.generate_dist_sequences(data, visit_sequences)



# generate time sequence

time_sequences = Helper.generate_time_sequences(data, visit_sequences)



# generage Type sequence

type_sequences = Helper.generate_type_sequence(data, visit_sequences)



# generate category sequence

cat_sequences, cat_reIndex_mapping = Helper.generate_cat_sequences(data, visit_sequences)


# generate ground truth for each sequence

ground_truth_sequences = Helper.generate_ground_truth_sequences(data, ground_truth_dict, POI_reIndex_mapping)



# generate specific poi ground truth for each sequence

specific_poi_sequences = Helper.generate_specific_poi_sequences(data, ground_truth_dict)


# ## 4. Extra Data Preperation

# ### Collective POI's category distribution
# 
# For each collective POI, count the number stores belongs to each category it has.
# The distribution is recorded in a 2-layer dictionary of form:
# 
# { POI_id (new id) : { category_id (new id): store count (int)} }



# generate collective POI's category distribution

poi_cat_distrib = Helper.generate_cat_distrib(data, valid_visits, POI_reIndex_mapping, cat_reIndex_mapping)


Helper.peep_dictionary(poi_cat_distrib)


valid_visit_data = data[data.index.isin(valid_visits)]


# ### Negative Samples for Each Sequence
# 
# For each user's each sequence, generate 'neg_sample_num' number of negative POIs
# 
# Negative POIs statisfy following criteria:
# 
# 1. The POI does not appear in the true sequence 
# 
# 2. The distance between:
#     *a) negative POI and true destination* and 
#     *b) true second last POI and true destination*
#    should be as close as possible
#    
# The output neg_sequences should be a 3d array of shape [user, seq, neg_sample]


# store distance between each valid POI (time consuming)
    
dist_mat = Helper.generate_POI_dist_mat(data, POI_reIndex_mapping)



# generate negative samples 

neg_sequences = Helper.generate_neg_sequences(POI_sequences, dist_mat, neg_sample_num, data, POI_reIndex_mapping, cat_reIndex_mapping)



# generate poi_cat_specific_poi_dict mapping

grouped = data.groupby(['POI_id', 'L2_id'])['Location_id'].unique().apply(list)



# generate poi_cat_specific_poi_dict

poi_cat_specific_poi_dict = {}

prev_poi = grouped.index[0][0]

cat_specific_poi_dict = {}

cat_specific_poi_dict[grouped.index[0][1]] = grouped[grouped.index[0]]

for index in grouped.index:

    if index[0] not in poi_cat_specific_poi_dict.keys():  
        
        poi_cat_specific_poi_dict[prev_poi] = cat_specific_poi_dict
        
        cat_specific_poi_dict = {}
        
        prev_poi = index[0]
        
        poi_cat_specific_poi_dict[index[0]] = {}
        
    cat_specific_poi_dict[index[1]] = grouped[index]
    
poi_cat_specific_poi_dict[prev_poi] = cat_specific_poi_dict


poi_cat_specific_poi_dict[317]


# ## 5. Form Sample Sets
# 
# Concatenate five sequences to form a sample, which is a tuple consists of: (POI_seq, dist_seq, time_seq, type_seq, cat_seq, neg_samplw)
# 
# Organise samples according to users in a dictionary of form:
# 
# { User_id (new id) : sample sets } 


# form sample set for each user

sample_sets = Helper.form_sample_sets(POI_sequences, dist_sequences, time_sequences, type_sequences, cat_sequences, ground_truth_sequences, specific_poi_sequences, neg_sequences)



Helper.peep_dictionary(sample_sets)


# # 6. Output Files


# set output directory

dir = './np_save_CHA/'
if small_sample:
    dir = './test_np_save_CHA/'



# create directory if not exists

if not os.path.exists(dir):
    os.makedirs(dir)



# save concatenated samples

Helper.save_dict(sample_sets, dir + 'sample_sets.pkl')



# save id mappings

np.save(dir + 'POI_reIndex_mapping.npy', POI_reIndex_mapping)
np.save(dir + 'user_reIndex_mapping.npy', user_reIndex_mapping)
np.save(dir + 'cat_reIndex_mapping.npy', cat_reIndex_mapping)


# save collective POI's category distribution dictionary

Helper.save_dict(poi_cat_distrib, dir + 'poi_cat_distrib.pkl')
Helper.save_dict(poi_cat_specific_poi_dict, dir + 'poi_cat_specific_poi_dict.pkl')



# save POI distance matrix 

np.save(dir + 'dist_mat.npy', dist_mat)



# save other relavant parameters

np.save(dir + 'max_dist.npy', max_dist) # max distance (for distance embedding)
np.save(dir + 'max_seq_len.npy', max_seq_len) # max sequence length (for input size)
np.save(dir + 'neg_sample_num.npy', neg_sample_num) # number of negative samples (for negative input size)

