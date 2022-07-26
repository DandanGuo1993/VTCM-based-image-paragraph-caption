from __future__ import print_function
import sys, os
import json

data = './data/captions/'

# Files
example_json = os.path.join(data, 'captions_val2014.json')
para_json = os.path.join(data, 'paragraphs_v1.json')
train_split = os.path.join(data, 'train_split.json')
val_split = os.path.join(data, 'val_split.json')
test_split = os.path.join(data, 'test_split.json')

# Output files (should be coco-caption directory)
coco_captions = './annotations/'
train_outfile = os.path.join(coco_captions, 'para_captions_train.json')
val_outfile = os.path.join(coco_captions, 'para_captions_val.json')
test_outfile = os.path.join(coco_captions, 'para_captions_test.json')
assert os.path.exists(coco_captions)

example_json = json.load(open(example_json))
para_json = json.load(open(para_json))
train_split = json.load(open(train_split))
val_split = json.load(open(val_split))
test_split = json.load(open(test_split))


train = {
    'images': [],
    'annotations': [],
    'info': {'description': 'Visual genome paragraph dataset (train split)'},
    'type': 'captions',
    'licenses': 'http://creativecommons.org/licenses/by/4.0/',
}

val = {
    'images': [],
    'annotations': [],
    'info': {'description': 'Visual genome paragraph dataset (val split)'},
    'type': 'captions',
    'licenses': 'http://creativecommons.org/licenses/by/4.0/',
}

test = {
    'images': [],
    'annotations': [],
    'info': {'description': 'Visual genome paragraph dataset (test split)'},
    'type': 'captions',
    'licenses': 'http://creativecommons.org/licenses/by/4.0/',
}

unique_ids = []
for imgid, item in enumerate(para_json):

    if imgid % 1000 == 0:
        print('{}/{}'.format(imgid, len(para_json)))

    url           = item['url']                   
    filename      = item['url'].split('/')[-1]     
    id            = item['image_id']              
    raw_paragraph = item['paragraph']
    split         = train if id in train_split else (val if id in val_split else test)

    if id in unique_ids: 
        continue
    else:
        unique_ids.append(id)

    image = {
        'url': item['url'],
        'file_name': filename,
        'id': id,
    }

    annotation = {
        'image_id': id,
        'id': imgid,
        'caption': raw_paragraph.replace(u'\u00e9', 'e') 
    }

    split['images'].append(image)
    split['annotations'].append(annotation)

print('Finished converting to coco-captions format.')
print('There are {} duplicate captions.'.format(len(para_json) - len(unique_ids)))
print('The {} split contains {} images and {} annotations'.format('train', len(train['images']), len(train['annotations'])))
print('The {} split contains {} images and {} annotations'.format('val', len(val['images']), len(val['annotations'])))
print('The {} split contains {} images and {} annotations'.format('test', len(test['images']), len(test['annotations'])))

for split, fname in [(train, train_outfile), (val, val_outfile), (test, test_outfile)]:
    with open(fname, 'w') as f:
        json.dump(split, f)


index_list = [val['annotations'][i]['image_id'] for i in range(len(val['annotations']))]
A = val['annotations'][index_list.index(2365490)]
