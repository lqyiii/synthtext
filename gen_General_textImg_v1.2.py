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


def length(line_vec):
    return float(np.dot(np.array(line_vec), np.array(line_vec).T)) ** 0.5

def perspective(img, pts1, pts2):
    print pts1
    print pts2
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    M = cv2.getPerspectiveTransform(pts1, pts2)

    cols, rows = pts2[-1]
    dst = cv2.warpPerspective(img, M, (cols, rows))
    return dst

def text_positions_transfer(wordBB):
    text_position = []
    for i in xrange(wordBB.shape[-1]):
        bb = wordBB[:, :, i]
        bb = np.c_[bb, bb[:, 0]]
        # plt.plot(bb[0, :], bb[1, :], 'g', alpha=alpha)
        tmp_line = []
        for j in range(4):
            tmp_line += [bb[0, :][j], bb[1, :][j]]
        text_position.append(tmp_line)

    return text_position

def intersection_angle(vec1, vec2):
    A_dot_B = np.dot(vec1, vec2.T)
    AB_module = (np.float(np.dot(vec1, vec1.T)) ** 0.5) * (np.float(np.dot(vec2, vec2.T)) ** 0.5)
    radius = acos(np.float(A_dot_B) / (AB_module + 0.0001))
    return abs(radius * 180 / pi)

def calc_quad_height(pts):
    pt1, pt2, pt3, pt4 = pts
    vec1 = np.float32(pt3) - np.float32(pt1)
    vec2 = np.float32(pt4) - np.float32(pt3)
    angle = intersection_angle(vec1, vec2)
    return sin(pi * angle / 180) * length(vec1)

def extend_randomly_bounding_box(pt_list, max_w_extension_rate=0.2, max_h_extension_rate=0.35):
    pt1 = (pt_list[0], pt_list[1]) # (x,y)
    pt2 = (pt_list[2], pt_list[3])
    pt3 = (pt_list[4], pt_list[5])
    pt4 = (pt_list[6], pt_list[7])

    left_w_extension_ratio = np.random.random()*max_w_extension_rate
    rigt_w_extension_ratio = np.random.random()*max_w_extension_rate

    upper_h_extension_ratio = np.random.random()*max_h_extension_rate
    lower_h_extension_ratio = np.random.random()*max_h_extension_rate

    pt1_left_extension = (pt1[0]-left_w_extension_ratio*(pt2[0]-pt1[0]), pt1[1]-left_w_extension_ratio*(pt2[1]-pt1[1]))
    pt2_rigt_extension = (pt2[0]+rigt_w_extension_ratio*(pt2[0]-pt1[0]), pt2[1]+rigt_w_extension_ratio*(pt2[1]-pt1[1]))

    pt3_rigt_extension = (pt3[0]+rigt_w_extension_ratio*(pt3[0]-pt4[0]), pt3[1]+rigt_w_extension_ratio*(pt3[1]-pt4[1]))
    pt4_left_extension = (pt4[0]-left_w_extension_ratio*(pt3[0]-pt4[0]), pt4[1]-left_w_extension_ratio*(pt3[1]-pt4[1]))

    pt1_upper_extension = (pt1[0]-upper_h_extension_ratio*(pt4[0]-pt1[0]), pt1[1]-upper_h_extension_ratio*(pt4[1]-pt1[1]))
    pt4_lower_extension = (pt4[0]+lower_h_extension_ratio*(pt4[0]-pt1[0]), pt4[1]+lower_h_extension_ratio*(pt4[1]-pt1[1]))

    pt2_upper_extension = (pt2[0]-upper_h_extension_ratio*(pt3[0]-pt2[0]), pt2[1]-upper_h_extension_ratio*(pt3[1]-pt2[1]))
    pt3_lower_extension = (pt3[0]+lower_h_extension_ratio*(pt3[0]-pt2[0]), pt3[1]+lower_h_extension_ratio*(pt3[1]-pt2[1]))

    pt1_new = RendererV3.intersection_point(line1=[pt1_left_extension,pt4_left_extension],line2=[pt1_upper_extension,pt2_upper_extension])
    pt2_new = RendererV3.intersection_point(line1=[pt2_rigt_extension,pt3_rigt_extension],line2=[pt1_upper_extension,pt2_upper_extension])
    pt3_new = RendererV3.intersection_point(line1=[pt2_rigt_extension,pt3_rigt_extension],line2=[pt3_lower_extension,pt4_lower_extension])
    pt4_new = RendererV3.intersection_point(line1=[pt1_left_extension,pt4_left_extension],line2=[pt3_lower_extension,pt4_lower_extension])

    return map(int,pt1_new+pt2_new+pt3_new+pt4_new)

def synthtext_part(img,wordBB):
    H,W = img.shape[:2]
    text_position=text_positions_transfer(wordBB)

    idx = 0
    partImg_info={}
    rand_x=np.random.randint(0,4)
    rand_y=np.random.randint(0,3)
    for rec in text_position:
        if min(rec)<0 or max(rec[0::2])>W or max(rec[1::2])>H:
            partImg_info[idx] = {'img':None,'height':None}
            idx+=1
            continue

        rec = extend_randomly_bounding_box(rec)
        pt1 = (max(rec[0],0), max(rec[1],0))
        pt2 = (min(rec[2],W), max(rec[3],0))
        pt3 = (min(rec[4],W), min(rec[5],H))
        pt4 = (max(rec[6],0), min(rec[7],H))
        pts1 = [pt1, pt2, pt4, pt3]
        w = length(np.float32(pt2)-np.float32(pt1))
        h = length(np.float32(pt1)-np.float32(pt4))
        pts2 = [[0,0],[w,0],[0,h],[w,h]]
        partImg = perspective(img,pts1,pts2)
        partImg_info[idx] = {'img':partImg,'height':calc_quad_height(pts1)}
        idx+=1
        # plt.imshow(partImg), plt.colorbar(), plt.show()
    return partImg_info

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
    synthtext_imgs_info=synthtext_part(img.copy(),wordBB)
    for i in range(len(synthtext_imgs_info)):
      img_info=synthtext_imgs_info[i]
      part_img = img_info['img']
      img_height = img_info['height']
      h,w = 0,0
      if part_img is not None:
          h,w = part_img.shape[:2]
      word=word_list[i]
      if part_img is not None and img_height>=18 and w/h>=0.7*len(word) and part_img.shape[0]>10 and part_img.shape[1]>10:
        cv.imwrite(img_filepath+imgname+'_'+str(i)+'_'+ticks+'.jpg',cv.cvtColor(part_img, cv.COLOR_RGB2BGR))
        txtfile.write(word+'\t'+imgname+'_'+str(i)+'_'+ticks+'.jpg'+'\r\n')

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


def main(generate_count=50,
         text_source_path=osp.join(DATA_PATH,'text_source/'),
         img_txt_filename='./data/words/label_general_imgs_190829_v1.txt',
         img_folder='./data/general_imgs_190829_v1/'):

  ticks_str=str(int(time.time()))
  # open databases:
  db_depth = h5py.File('./data/download/depth.h5', 'r')
  db_seg = h5py.File('./data/download/seg.h5', 'r')
  imnames = pkl.load(open('./data/download/imnames.cp', 'r'))
  
  # open the output h5 file:
  txtfile=codecs.open(img_txt_filename,'aw+','utf-8')
  out_db = h5py.File(OUT_FILE,'w')
  out_db.create_group('/data')
  print colorize(Color.GREEN,'Storing the output in: '+OUT_FILE, bold=True)

  RendererV3.intersection_point()

  RV3 = RendererV3(DATA_PATH,
                   max_time=SECS_PER_IMG,
                   TextPath=text_source_path,
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
                  #add_res_to_disk(img_name, deepcopy(res), img_folder, txtfile)
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
    main(generate_count=50,
         text_source_path=osp.join(DATA_PATH,'text_source_0829/'),
         )