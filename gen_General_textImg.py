# -*- coding: utf-8 -*-
# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""

import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
from synthgen import *
from common import *
import cv2 as cv
import time
from math import *
import codecs
from copy import deepcopy
import cPickle as pkl

def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    imgOut = imgRotation[int(pt1[1]):int(pt3[1]), int(pt1[0]):int(pt3[0])]
    height, width = imgOut.shape[:2]
    return imgOut


def text_positions_transfer(wordBB):
    text_position=[]
    for i in xrange(wordBB.shape[-1]):
        bb = wordBB[:, :, i]
        bb = np.c_[bb, bb[:, 0]]
        #plt.plot(bb[0, :], bb[1, :], 'g', alpha=alpha)
        tmp_line=[]
        for j in range(4):
            tmp_line+=[bb[0, :][j],bb[1, :][j]]
        text_position.append(tmp_line)

    return text_position


def synthtext_part(img,wordBB):
    text_position=text_positions_transfer(wordBB)
    partImgs=[]
    rand_x=np.random.randint(0,4)
    rand_y=np.random.randint(0,3)
    for rec in text_position:
        pt1 = (rec[0]-rand_x, rec[1]-rand_y)
        pt2 = (rec[2]+rand_x, rec[3]-rand_y)
        pt3 = (rec[4]+rand_x, rec[5]+rand_y)
        pt4 = (rec[6]-rand_x, rec[7]+rand_y)
        partImg = dumpRotateImage(img.copy(), degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])), pt1, pt2, pt3, pt4)
        # plt.imshow(partImg), plt.colorbar(), plt.show()
        partImgs.append(partImg)
    return partImgs

## Define some configuration variables:
NUM_IMG = -1 # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 1 # no. of times to use the same image
SECS_PER_IMG = 10 #max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
DB_FNAME = osp.join(DATA_PATH,'dset.h5')
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_FILE = 'results/SynthText_cartoon_viz.h5'

TEXTPATH = osp.join(DATA_PATH,'text_source/')

def add_res_to_db(imgname,res,db):
  """
  Add the synthetically generated text image instance
  and other metadata to the dataset.
  """
  ninstance = len(res)
  for i in xrange(ninstance):
    print colorize(Color.GREEN,'added into the db %s '%res[i]['txt'])
    
    dname = "%s_%d"%(imgname, i)
    db['data'].create_dataset(dname,data=res[i]['img'])
    db['data'][dname].attrs['charBB'] = res[i]['charBB']
    db['data'][dname].attrs['wordBB'] = res[i]['wordBB']
    print 'type of res[i][\'txt\'] ',type(res[i]['txt'])
         
    #db['data'][dname].attrs['txt'] = res[i]['txt']
    db['data'][dname].attrs.create('txt', res[i]['txt'], dtype=h5py.special_dtype(vlen=unicode))
    print 'type of db ',type(db['data'][dname].attrs['txt']) 
    print colorize(Color.GREEN,'successfully added')
    print res[i]['txt']
    print res[i]['img'].shape
    print 'charBB',res[i]['charBB'].shape
    print 'charBB',res[i]['charBB']
    print 'wordBB',res[i]['wordBB'].shape
    print 'wordBB',res[i]['wordBB']
    '''
    img = Image.fromarray(res[i]['img'])
    hsv_img=np.array(rgb2hsv(img))
    print 'hsv_img_shape',hsv_img.shape
    print 'hsv_img',hsv_img
    H=hsv_img[:,:,2]
    print 'H_channel',H.shape,H
    #img = Image.fromarray(db['data'][dname][:])
    '''

def add_res_to_disk(imgname,res,img_filepath,txtfile):
  ninstance = len(res)
  ticks=str(int(time.time()))
  for i in xrange(ninstance):
    img=res[i]['img']
    wordBB=res[i]['wordBB']
    word_tmp=res[i]['txt']
    word_list=[]
    for i in word_tmp:
        word_list+=i.split('\n')
    synthtext_imgs=synthtext_part(img.copy(),wordBB)
    for i in range(len(synthtext_imgs)):
      part_img=synthtext_imgs[i]
      word=word_list[i]
      if part_img.shape[0]>10 and part_img.shape[1]>10:
        cv.imwrite(img_filepath+imgname+'_'+str(i)+'_'+ticks+'.jpg',cv.cvtColor(part_img, cv.COLOR_RGB2BGR))
        txtfile.write(word+','+imgname+'_'+str(i)+'_'+ticks+'.jpg'+'\r\n')

def rgb2hsv(image):
    return image.convert('HSV')

def rgb2gray(image):
    
    rgb=np.array(image)
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# 随机选择图片和seg以及depth
def random_choose(imnames,db_seg,db_depth,max_index=8009):
    while True:
        rand_v=np.random.randint(max_index)
        img_name=imnames[rand_v]
        img=cv.imread('./data/download/bg_img/'+img_name)
        if img is not None:
            break
    seg=db_seg['mask'][img_name]
    depth=db_depth[img_name]
    RGB_img=cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rgb_img=Image.fromarray(RGB_img)
    return rgb_img,seg,depth,img_name


def main(generate_count=10):
  ticks_str=str(int(time.time()))
  # open databases:
  db_depth = h5py.File('./data/download/depth.h5', 'r')
  db_seg = h5py.File('./data/download/seg.h5', 'r')
  imnames = pkl.load(open('./data/download/imnames.cp', 'r'))
  
  # open the output h5 file:
  img_folder = './data/general_imgs/'
  txtfile=codecs.open('./data/words/label_nums_'+ticks_str+'.txt','aw+','utf-8')
  out_db = h5py.File(OUT_FILE,'w')
  out_db.create_group('/data')
  print colorize(Color.GREEN,'Storing the output in: '+OUT_FILE, bold=True)

  RV3 = RendererV3(DATA_PATH,
                   max_time=SECS_PER_IMG,
                   TextPath=TEXTPATH,
                   min_char_height=16,
                   max_text_regions=10,
                   min_nchar=2,
                   min_font_h=24,
                   max_font_h=36
                   )

  for i in range(generate_count):
      try:

          if i%100==0:
              print 'processing count:',i
          t1=time.time()
          img, seg_, depth_, img_name=random_choose(imnames,db_seg,db_depth)
          img_resize = img.resize(depth_.shape[-2:])
          depth=depth_[:].T

          hsv_img = np.array(rgb2hsv(img_resize))
          H = hsv_img[:, :, 2]
          H = H.T
          H = H.astype('float32')

          gray = np.array(rgb2gray(img_resize))
          # print 'gray',gray.shape,gray
          depth = (np.max(depth) / np.max(gray)) * gray.astype('float32')
          kernel = np.ones((5, 5), np.float32) / 25
          gray = cv2.filter2D(gray, -1, kernel)

          # get segmentation:
          seg = seg_[:].astype('float32')
          area = seg_.attrs['area']
          label = seg_.attrs['label']

          sz = depth.shape[:2][::-1]
          img = np.array(img.resize(sz, Image.ANTIALIAS))
          seg = np.array(Image.fromarray(seg).resize(sz, Image.NEAREST))

          res = RV3.render_text(img, depth, seg, area, label,
                                ninstance=INSTANCE_PER_IMAGE, viz=False)
          t2 = time.time()

          for ct in range(5):

              if len(res) > 0:
                  # non-empty : successful in placing text:
                  add_res_to_db(img_name, res, out_db)
                  add_res_to_disk(img_name, deepcopy(res), img_folder, txtfile)
                  break
              else:
                  res = RV3.render_text(img, depth, seg, area, label,
                                        ninstance=INSTANCE_PER_IMAGE, viz=False)


          # visualize the output:
          print 'time consume in each pic', t2 - t1
      except:
          print traceback.format_exc()

  out_db.close()


if __name__=='__main__':
    main()