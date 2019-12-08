import warnings
warnings.filterwarnings(action='ignore')
import os
import sys
import keras
from utils import rle2mask, img_preprocess
from keras.models import load_model
from model import *
from generator import * 
from utils import *
# from metric import bce_dice_loss,bce_logdice_loss,dice_coef,dice_loss
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
import sys
import argparse
import time
from data import df_gen
import albumentations as albu
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import gc

from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)




FLAGS = None
# print('!!!')
# print(FLAGS)
# print(FLAGS.proba_thres)
# label_proba = [float(k) for k in FLAGS.proba_thres]
# print(label_proba)
# size_thres = [int(k) for k in FLAGS.size_thres]
# pad_thres = [int(k) for k in FLAGS.pad_thres]
# convex_hull = [bool(k) for k in FLAGS.convex_hull]



    
# class TestPredictor:

def predict_df(img_path, origin_size = (1200,1600),label_name=['FISH', 'FLOWER','SUGAR', 'GRAVEL']):
    img_set = [k for k in os.listdir(img_path) if k.split('.')[-1]==('jpg' or 'png' or 'jpeg')]
    df = pd.DataFrame(img_set, columns=['ImageId'])
    
    lable_nan_arr = np.empty((df.shape[0], len(label_name))) 
    lable_nan_arr[:] = np.nan
    df_label = pd.DataFrame(lable_nan_arr, columns = label_name)
    
    origin_size_nam_arr = np.empty((df.shape[0], 1))
    origin_size_nam_arr[:] = np.nan
    df_origin_size = pd.DataFrame(origin_size_nam_arr, columns = ['size'])
    
    colors_nan = np.empty((df.shape[0],1))
    colors_nan[:] = np.nan
    df_colors = pd.DataFrame(colors_nan, columns = ['colors'])
    
    df = pd.concat([df,df_origin_size,df_colors,df_label],axis=1)
    
    return df


def batch_predict(model, test_df, origin_size , colors, label_names ,pred_batch_size = 10, ):
    batch_res_df = []
    
    
    
    
    for i in tqdm(range(0, test_df.shape[0], pred_batch_size)):
        batch_idx = list(range(i, min(test_df.shape[0], i + pred_batch_size)))
        
        batch_generator  = SegmentDataGenerator( test_df.iloc[batch_idx], subset='test', batch_size = 1,
                                               shuffle=False, preprocess = img_preprocess,
                                               augmentation = None, resize_shape = FLAGS.resize_shape,
                                               test_path = FLAGS.img_path)
        
        
        batch_preds = model.predict_generator(batch_generator, verbose = 1)
        batch_preds_re = predict_resize(batch_preds, 
                                       proba=[float(k) for k in FLAGS.proba_thres], 
                                       pad_size=[int(k) for k in FLAGS.pad_thres], 
                                       reduce_size = [int(k) for k in FLAGS.size_thres], 
                                       convex= [bool(k) for k in FLAGS.convex_hull],
                                        origin_img_size = (origin_size[0],origin_size[1])
                                     )
        # np.save('../test.npy',batch_preds_re)
        # sys.exit()
        
        for j, b in enumerate(batch_idx):
            filename = test_df['ImageId'].iloc[b]
            image_df = test_df[test_df['ImageId'] == filename].copy()
            # print(image_df)
            preds_mask = batch_preds_re[j]
            
            # print(origin_size)
            image_df['size'] = [[origin_size[0], origin_size[1]]]
            image_df['colors'] = [colors]
            # print(image_df)
            # sys.exit()
            for l_idx, label in enumerate(label_names):
                label_rle = mask2rle(preds_mask[:,:,l_idx])
                # print(label_rle)

                image_df[label] = label_rle['counts']
            batch_res_df.append(image_df)
            gc.collect()
        gc.collect()
    batch_res_df = pd.concat(batch_res_df)
    
    print(batch_res_df)
    # sys.exit()
    batch_res_df.replace(to_replace=[None], value = np.nan, inplace = True)
    print(batch_res_df)
    print(batch_res_df.shape)
    print('Batch predict end, Images : {}, labels : {}'.format(batch_res_df.shape[0], batch_res_df.shape[1]-2))
    
    
    
    return batch_res_df


    




def main():
    # label_proba = [float(k) for k in FLAGS.proba_thres]
    # size_thres = [int(k) for k in FLAGS.size_thres]
    # pad_thres = [int(k) for k in FLAGS.pad_thres]
    # convex_hull = [bool(k) for k in FLAGS.convex_hull]
    
     
    # print(label_proba)
    # sys.exit()
    df = df_gen(FLAGS.json_path)
    origin_shape = df.iloc[0]['size']
    colors = df.iloc[0]['colors']
    print(origin_shape)
    print(colors)
    # sys.exit()
    print(df.keys())
    labels = df.keys()[3:].values.tolist()
    print(labels)
    # sys.exit()
    
    print(FLAGS.model_path)
    # sys.exit()
    K.clear_session()
    model = load_model(FLAGS.model_path, custom_objects = {'tf' : tf, 
                                                           'bce_dice_loss' : bce_dice_loss,
                                                          'dice_coef' : dice_coef})
    print(model)
    # sys.exit()
    
    df_test = predict_df(FLAGS.img_path, origin_size = origin_shape ,label_name=labels)
    print(df_test)
    # sys.exit()
    pred_test = batch_predict(model, df_test, origin_size = origin_shape , label_names = labels ,pred_batch_size = FLAGS.batch_size, colors =colors)
    
    pred_path = '/'.join(FLAGS.model_path.split('/')[:-2]) +  '/pred/'
    
    pred_test.to_csv(os.path.join(pred_path, 'batch_pred.csv'), index=False)
    print(pred_test)
    # sys.exit()
    
    for r_idx in tqdm(range(pred_test.shape[0])):
        
        
        img = rle_mask2img(pred_test.iloc[r_idx], FLAGS.img_path)
        print(pred_path)
        # print( pred_test.iloc[r_idx]['ImageId'].split('.')[0])
        # print(pred_test.iloc[r_idx]['ImageId'].split('.')[1])
        # print('!!!!')
        print(pred_test.iloc[r_idx]['ImageId'].split('.')[0] + '_segment.' + pred_test.iloc[r_idx]['ImageId'].split('.')[1])
        save_path = os.path.join(pred_path, pred_test.iloc[r_idx]['ImageId'].split('.')[0] + '_segment.' + pred_test.iloc[r_idx]['ImageId'].split('.')[1])
        img.save(save_path)
        print ('Segment {} done !!! '.format(pred_test.iloc[r_idx]['ImageId']))
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'testing')
    parser.add_argument('--method', required=True, help = 'batch  or realtime')
    # parser.add_argument('--train_path')
    # parser.add_argument('--test_path')
    parser.add_argument('--json_path', required=True)
    parser.add_argument('--img_path', required=True)
    parser.add_argument('--model_path',required = True)
    parser.add_argument('--proba_thres', nargs='+', type = float, help = 'Post - label proba', default = [0.5])
    parser.add_argument('--size_thres', nargs='+', type = int, help = 'Post - label size', default = [3000])
    parser.add_argument('--pad_thres', nargs='+', type = int, help = 'Post - label pad size', default = [3])
    parser.add_argument('--convex_hull',  nargs='+',type = bool,  help = 'Post - label convex hull', default = [True])
    # parser.add_argument('--epoch', default = 50, type =int)
    # parser.add_argument('--batch_size', default = 16, type =int)
    
    # parser.add_argument('--resize_shape', default = (256,256))
    # parser.add_argument('--origin_shape', default = (1200,1200))
    parser.add_argument('--resize_shape', nargs='+', type=int, default = [256,256])
    parser.add_argument('--origin_shape', nargs='+', type=int, default = [1200,1400])
    parser.add_argument('--batch_size', default = 10)
    # parser.add_argument('--augument',default = True)
    
    
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    print(FLAGS.proba_thres)
    
    
    
    main()
        
        
















# if __name__ == '__main__':
    
#     parser = argparse.ArgumentParser(description = 'predict')
#     # parser.add_argument('--target', required=True, help = 'train or predict')
#     # parser.add_argument('--train_path')
#     # parser.add_argument('--test_path')
#     parser.add_argument('--json_path', required=True)
#     parser.add_argument('--img_path', required=True)
#     parser.add_argument('--model_path',required=True)
#     # parser.add_argument('--epoch', default = 50, type =int)
#     # parser.add_argument('--batch_size', default = 16, type =int)
    
#     parser.add_argument('--resize_shape', default = (256,256))
#     # parser.add_argument('--augument',default = True)
    
    
#     FLAGS, unparsed = parser.parse_known_args()
    
    
#     main()