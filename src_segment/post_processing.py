import pandas as pd
import numpy as np
from utils import *
from metrics import *
from multiprocessing import Pool
import cv2
from tqdm import tqdm
from functools import partial


param_grid = {'proba' : np.round(np.array([0,0.25,0.50,0.75, 1.0],dtype =np.float64),2),
                   'pad' : np.array([0,3,5,10]),
                   'reduce_size' : np.arange(5000,20000,5000,dtype=np.int64),
#               'convex': [True, False]
              
 }



def predict_post_grid(msks, proba = 0.9, pad = False, pad_size = 10, reduce = False, reduce_size =10000, convex = False, origin_size = [1400,2100] ):
#     print(msks.shape)

    resized_msks = np.zeros((msks.shape[0],origin_size[0],origin_size[1]))
    for i in range(len(msks)):
#         cls = msks[1][i]
#         msk = msks[0][i,:,:] * cls
        msk = msks[i,:,:]
        msk = np.array(msk >=proba, dtype = np.uint8)
        msk = cv2.resize(msk,(origin_size[1],origin_size[0]), interpolation = cv2.INTER_LINEAR)

        msk = mask2pad(msk, pad_size)

        msk = masks_reduce(msk, reduce_size)

        msk = contour_convexHull(msk.astype(np.uint8))
        
        resized_msks[i, : , :] = msk
    return resized_msks




def paralleize_numpy(preds, func ,proba = 0.9, pad_size = 10, reduce_size = 10000, convex =False, origin_size=[1400,2100], cores=2):
    np_split = np.array_split(preds, cores)
#     np_split_cls = np.array_split(preds_cls, cores)
    
#     value = list(zip(np_split, np_split_cls))
    
    pool = Pool(cores)
    res_np = np.concatenate(pool.map(partial(func, proba = proba,pad_size = pad_size, reduce_size = reduce_size , convex = convex, origin_size = origin_size), np_split ))
    pool.close()
    pool.join()
    return res_np





def post_optimi(valid_true, valid_pred, origin_size=[1400,2100], lables =['fish', 'flower', 'gravel', 'sugar']):
    val_dice_max = []
    val_dice_tabel = pd.DataFrame(columns = ['label', 'proba', 'reduce_size', 'convex', 'dice'])
    
    for l_idx label in enumerate(labels):
        for p_idx,para in tqdm(enumerate(parameter)):
            
            valid_preds_pp = paralleize_numpy(valid_pred[:,:,:,l_idx],
                                          predict_post_grid, 
                                          proba=para['proba'], 
                                          pad_size = para['pad'], 
                                          reduce_size = para['reduce_size'],
                                         cores = 2,
                                         convex=para['convex'],
                                             origin_size = origin_size)
            dice_score = dice_channel_label(valid_preds_pp, valid_true[:,:,:, l_idx])

            val_dice_tabel = val_dice_tabel.append({'label' : label,
                                                   'proba' : para['proba'],
                                                   'reduce_size' : para['reduce_size'],
                                                    'convex': para['convex'],
                                                   'dice' : dice_score},
                                                   ignore_index=True)


            gc.collect()
        label_max_table = val_dice_tabel[val_dice_tabel['label']==label]
        print(label_max_table[label_max_table['dice'] == tt['dice'].max()])
        val_dice_max.append(label_max_table)
        
        gc.collect()
    return val_dice_max
    # val_dice_table_ensemble.to_csv('val_dice_table_ensemble_2.csv')
