from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import h5py
import argparse
import numpy as np
parser = argparse.ArgumentParser()

train_output_file = h5py.File('./data/feature_from_dense_cap/im2p_train_output.h5', 'r')
train_feats = train_output_file.get('feats').value
train_boxes = train_output_file.get('boxes').value
train_feats_average = np.mean(train_feats,axis=1)
f = open('./data/feature_from_dense_cap/imgs_train_path.txt',"r")
lines = f.readlines()
train_image_id = []
for line in lines:
    train_image_id.append(line[25:-5])
for i in range(len(train_feats)):
    np.savez_compressed(os.path.join('./data/densecaption_att', train_image_id[i]), feat=train_feats[i])
    np.save(os.path.join('./densecaption_fc', train_image_id[i]), train_feats_average[i])
    np.save(os.path.join('./densecaption_box', str(train_image_id[i])), train_boxes[i])
    print("train_data:",i)

val_output_file = h5py.File('./data/feature_from_dense_cap/im2p_val_output.h5', 'r')
val_feats = val_output_file.get('feats').value
val_boxes = val_output_file.get('boxes').value
val_feats_average = np.mean(val_feats,axis=1)
f = open('./data/feature_from_dense_cap/imgs_val_path.txt',"r")
lines = f.readlines()
val_image_id = []

for line in lines:
    val_image_id.append(line[23:-5])

for i in range(len(val_feats)):
    np.savez_compressed(os.path.join('./data/densecaption_att', val_image_id[i]), feat=val_feats[i])
    np.save(os.path.join('./data/densecaption_fc', val_image_id[i]), val_feats_average[i])
    np.save(os.path.join('./data/densecaption_box', str(val_image_id[i])),  val_boxes[i])
    print("val_data:",i)

test_output_file = h5py.File('./data/feature_from_dense_cap/im2p_test_output.h5', 'r')
test_feats = test_output_file.get('feats').value
test_boxes = test_output_file.get('boxes').value
test_feats_average = np.mean(test_feats,axis=1)
f = open('./data/feature_from_dense_cap/imgs_test_path.txt',"r")
lines = f.readlines()
test_image_id = []
for line in lines:
    test_image_id.append(line[24:-5])
for i in range(len(test_feats)):
    np.savez_compressed(os.path.join('./data/densecaption_att', test_image_id[i]), feat=test_feats[i])
    np.save(os.path.join('./data/densecaption_fc', test_image_id[i]), test_feats_average[i])
    np.save(os.path.join('./data/densecaption_box', test_image_id[i]), test_boxes[i])
    print("test_data:",i)

for i in range(len(test_image_id)):
    if test_image_id[i] == '2383927':

        print(i)


for i in range(len(test_feats)):
    for j in range(50):
        if sum(test_feats[i][j]) <= 0:
            print (i,j)
print('end')