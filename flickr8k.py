import tensorflow as tf
import re
import numpy as np
import os
import time
import json
import pickle

# load doc into memory
def load_doc(filename):
  # open the file as read only
  file = open(filename, 'r')
  # read all text
  text = file.read()
  # close the file
  file.close()
  return text

# extract descriptions for images
def load_descriptions(doc, img_dir):
  mapping = dict()
  all_img_name_vector1 = []
  all_captions1 = []


  # process lines
  for line in doc.split('\n'):
    # split line by white space
    tokens = line.split()
    if len(line) < 2:
      continue
    # take the first token as the image id, the rest as the description
    image_id, image_desc = tokens[0], tokens[1:]
    # remove filename from image id
    image_id = image_id.split('.')[0]
    # convert description tokens back to string
    image_desc = ' '.join(image_desc)
    # create the list if needed
    if image_id not in mapping:
      mapping[image_id] = list()
    # store description
    mapping[image_id].append(image_desc)

    caption = '<start> ' + image_desc + ' <end>'
    if image_id ==  '2258277193_586949ec62':    
      pass
    else:
      full_coco_image_path = os.path.join(img_dir, '{}.jpg'.format(image_id))
      # full_coco_image_path = 'data/flickr8k/Flicker8k_Dataset/' + '%s.jpg' % (image_id)
      all_img_name_vector1.append(full_coco_image_path)
      all_captions1.append(caption)


  return all_captions1,all_img_name_vector1

def get_flickr8k_data():

    name_of_zip = 'Flickr8k_Dataset.zip'

    data_dir = os.path.join(os.path.abspath('.'), 'data/flickr8k')

    if not os.path.exists(data_dir + '/' + name_of_zip):
      image_zip = tf.keras.utils.get_file(name_of_zip, 
                                          cache_subdir=data_dir,
                                          origin = 'http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_Dataset.zip',
                                          extract = True)

    img_dir = data_dir + '/Flicker8k_Dataset/'

    name_of_zip = 'Flickr8k_text.zip'

    data_dir = os.path.join(os.path.abspath('.'), 'data/flickr8k')

    if not os.path.exists(data_dir + '/' + name_of_zip):
      image_zip = tf.keras.utils.get_file(name_of_zip, 
                                          cache_subdir=data_dir,
                                          origin = 'http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_text.zip',
                                          extract = True)



    all_captions1 = []
    all_img_name_vector1 = []
    filename = 'data/flickr8k/Flickr8k.token.txt'
    # load descriptions
    doc = load_doc(filename)
    # parse descriptions
    all_captions1,all_img_name_vector1 = load_descriptions(doc, img_dir)

    train_caption=[]
    img_name_vector=[]
    train_captions = all_captions1
    img_name_vector = all_img_name_vector1
    print('Captions loaded:', len(train_captions) )
    print('Images:', len(img_name_vector) )
    return train_captions,img_name_vector


def get_flickr30k_data():
    """
    Note: This Dataset is not automatically downloaded. The Following steps need to be done before this data
    set can be used:

    1. Get the dataset (images) from:
        https://drive.google.com/file/d/0B_PL6p-5reUAZEM4MmRQQ2VVSlk/view?usp=sharin

    2. Get the annotations from:
        http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/

        - Rename as - Flickr30k.token.txt
        - Add Flickr30k.token.txt to the folder created in step2.

    3.In data folder create a  Flickr30k_Dataset
      Extract the images from step1 in this directory - !tar xvzf flickr30k_images.tar.gz

    4. Copy all the images in flickr30k_images to the Flickr30k_Dataset directory

    :return: [captions, images_paths]
    """

    print("Getting Flicker30k Data ...")
    # name_of_zip = 'Flickr8k_Dataset.zip'

    data_dir = 'data/flickr30k/Flicker30k_Dataset'
    print("Data dir is", data_dir)

    captions_file = 'data/flickr30k/Flickr30k.token.txt'

    # load descriptions
    doc = load_doc(captions_file)
    # parse descriptions
    captions, images_paths = load_descriptions(doc, data_dir)

    print('Number of images {}. Number of captions {}'.format(len(images_paths), len(captions)))

    return captions, images_paths