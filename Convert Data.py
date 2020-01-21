# dependencies

import numpy as np
import pandas as pd
import collections


data1 = pd.read_csv('./1_all_useful_POI_Charlotte.csv', encoding='utf-8')
data2 = pd.read_csv('./yelp_foursquare_allSortedData_CHA.csv', encoding='ISO-8859-1')


data2['Time'] = data2['TimeStamp'].apply(lambda x: x.split()[3])

# create date and time columns

mon_dic = {'Jan':'1', 'Feb':'2', 'Mar':'3', 'Apr':'4', 'May':'5', 'Jun':'6', 'Jul':'7', 'Aug':'8', 'Sep':'9', 'Oct':'10', 'Nov':'11', 'Dec':'12'}

data2['Date'] = data2['TimeStamp'].apply(lambda x: x.split()[5] + mon_dic[x.split()[1]] + x.split()[2])


# Create local time column

data2['Local_sg_time'] = data2['Date'].map(str) + ' ' + data2['Time']
data2['Local_sg_time'] = data2['Local_sg_time'].apply(lambda x: x.split()[0][0:4]+'/'+x.split()[0][4:-2]+'/'+x.split()[0][-2:] + ' ' + x.split()[1])
data2['Local_sg_time'].apply(lambda x: x[:5]+'0'+x[5:] if len(x.split()[0].split('/')[1])==1 else x)

# create category id 

cat_counter = collections.Counter(data2['Category'])
cat_id_mapping = dict(zip(cat_counter.keys(), np.arange(len(cat_counter.keys()))))
data2['L2_id'] = data2['Category'].apply(lambda x: cat_id_mapping[x])

# merge to add rating information

data3 = pd.merge(data2, data1, how='left', on=['location_id'])


data3.columnsdata3.columns = ['Unnamed', 'Location_id', 'POI_id', 'POI_Type', 'Org_id', 'User_id', 'TimeStamp', 'Zone', 'Latitude', 'Longitude', 'Category_2', 'yelp_ID', 'name', 'stars', 'Time', 'date', 'Local_sg_time', 'L2_id', 'Unnamed: 0_y', 'POI_id_y', 'POI_Type_y', 'Latitude_y', 'Longitude_y', 'Category_2_y', 'Org_id_y', 'yelp_ID_y', 'name_y', 'stars_y']


data2.to_csv('data_CHA.csv')


