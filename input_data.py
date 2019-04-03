# adopted from https://github.com/hx173149/C3D-tensorflow/blob/master/input_data.py
# ==============================================================================

import os
import PIL.Image as Image
import random
import numpy as np
import cv2

def get_frames_data(filename, num_frames_per_clip=16, crop_size=112):
  ''' Given a directory containing extracted frames, return a video clip of
  (num_frames_per_clip) consecutive frames as a list of np arrays '''
  ret_arr = []
  s_index = 0
  np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
  for parent, dirnames, filenames in os.walk(filename):
    total_files = len(filenames)
    if(total_files<num_frames_per_clip):
        diff  = num_frames_per_clip - len(filenames)
        for k in range(diff):
            dup = random.randint(0, total_files + k - 1)
            filenames.insert(dup, filenames[dup])
    filenames = sorted(filenames)
    s_index = random.randint(0, len(filenames) - num_frames_per_clip)
    j = 0
    for i in range(s_index, s_index + num_frames_per_clip):
      image_name = str(filename) + '/' + str(filenames[i])
      img = Image.open(image_name)
      if(img.width>img.height):
        scale = float(crop_size)/float(img.height)
        img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
      else:
        scale = float(crop_size)/float(img.width)
        img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)
      crop_x = int((img.shape[0] - crop_size)/2)
      crop_y = int((img.shape[1] - crop_size)/2)
      img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] - np_mean[j]
      img_data = np.array(img)
      ret_arr.append(img_data)
      j+=1
    ret = np.array(ret_arr)
  return np.array(ret_arr)
