import warnings
warnings.filterwarnings(action='ignore')
import os
import sys
import keras
from utils import rle2mask, img_preprocess
from model import *
# from metric import bce_dice_loss,bce_logdice_loss,dice_coef,dice_loss
import numpy as np
import tensorflow as tf
import sys
import argparse
import time
from data import df_gen
import albumentations as albu
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)


now = time.gmtime(time.time())
local_time = '{}_{}_{}_{}_{}'.format(now.tm_year,now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
seg_time = 'seg_ment_{}_{}_{}_{}_{}'.format(now.tm_year,now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)



FLAGS = None



# def argument():
    
#     parser = argparse.ArgumentParser(description = 'testing')
#     parser.add_argument('--target', required=True, help = 'train or predict')
#     # parser.add_argument('--train_path')
#     # parser.add_argument('--test_path')
#     parser.add_argument('--json_path', required=True)
#     parser.add_argument('--model_name',default= local_time)
#     parser.add_argument('--epoch', default = 50)
#     parser.add_argument('--batch_size', default = 16)
    
#     parser.add_argument('--shape', default = (256,256))
#     parser.add_argument('--augument',default = True)
#     # parser.add_argument('--')
#     return parser
            

class DataGenerator(keras.utils.Sequence):
    def __init__(self, df, batch_size = 16 ,subset ='train', shuffle = False, preprocess = None, info={}, augmentation = None, resize_shape = (256,256), train_path=None, test_path=None, ):
        super().__init__()
        self.df = df
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.info = info
        self.augmentation = augmentation
        self.resize_shape = resize_shape
        
        if self.subset =='train':
            # self.data_path = path +'train_images/'
            self.data_path = train_path + '/'
        elif self.subset =='test':
            # self.data_path = path + 'test_images/'
            self.data_path = test_path + '/'
        self.on_epoch_end()
        
    def __len__(self):
        
        return int(np.floor(len(self.df) / self.batch_size))
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __random_transform__(self, img, masks):
        composed = self.augmentation(image = img, mask = masks)
        aug_img = composed['image']
        aug_masks = composed['mask']
        
        return aug_img, aug_masks
    def __getitem__(self,index):
        x = np.empty((self.batch_size, self.resize_shape[0], self.resize_shape[1], 3), dtype=np.float32)
        y = np.empty((self.batch_size, self.resize_shape[0], self.resize_shape[1], 4), dtype=np.int8)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        for i,f in enumerate(self.df['ImageId'].iloc[indexes]):
            self.info[index*self.batch_size + i] =f 
#             x[i,]=Image.open(self.data_path + f).resize((256,256), resample = Image.)
            
            # print (self.data_path + f)
            img_value=cv2.imread(self.data_path + f)
            x[i,] = cv2.resize(img_value,\
                               (self.resize_shape[1], self.resize_shape[0]),\
                               interpolation=cv2.INTER_AREA)
            
            
            
            
#             x[i,] = cv2.imread(self.data_path + f)(img_value, (256,256),interpolation=cv2.INTER_AREA)
            if self.subset =='train':
                # print(self.df.columns[2:].to_list())
                for j,label in enumerate(self.df.columns[2:].to_list()):
                    rle = self.df[label].iloc[indexes[i]]
#                     print(rle)
                    shape = self.df['size'].iloc[indexes[i]]
                    y[i,:,:,j] = rle2mask(rle, shape, resize_shape = self.resize_shape )
                    
            if not self.augmentation is None:
                x[i,], y[i,] = self.__random_transform__(x[i,], y[i,])
        if self.preprocess !=None : x= self.preprocess(x)
        if self.subset == 'train' : return x,y
        else: return x

def augumentation():
    augument = albu.Compose([albu.VerticalFlip(), 
                         albu.HorizontalFlip(), 
                         albu.Rotate(limit= 20),
                         albu.GridDistortion(),
#                     albu.RandomSizedCrop((128,174), 256, 384,interpolation =cv2.INTER_LINEAR )
                        ],p=1.0)
    return augument

def label_concat(df,labels):
    df = df[labels]
    label_list = df[df!=''].keys().values.tolist()
    return label_list

def train_split(df):
    df_copy = df.copy()
    df_copy['Class'] = df_copy.apply(lambda x : label_concat(x, df.columns[2:].tolist() ), axis=1)
    train_idx, valid_idx = train_test_split(df_copy.index.values,
                                            test_size =0.2,
                                            stratify = df_copy['Class'].map(lambda x : str(sorted(x))),
                                            random_state=42)
    return train_idx,valid_idx


def compile_and_train(model,train_batches,valid_batches, epochs, pretrained_weights = None, model_path ='./model_test/', graph_path = './graph_test/', log_path = './log/' ): 
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    model.compile(loss=bce_dice_loss, optimizer='adam', metrics=[dice_coef,bce_dice_loss])
    early_stopping = EarlyStopping(monitor='val_loss', mode = 'min',patience=20, verbose=1)
    ###=============
    MODEL_SAVE_FOLDER_PATH = model_path
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)
    model_path = MODEL_SAVE_FOLDER_PATH + 'effinet_b4_unet_{epoch:02d}-{val_loss:.4f}'
    ###=============
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', 
                                   mode = 'min', save_best_only=True, verbose=1)
    #model_checkpoint = ModelCheckpoint("./" + model_name + "_best.h5",monitor='val_acc', 
    #                               mode = 'max', save_weights_only=True, save_best_only=True, period=1, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode = 'min',factor=0.2, patience=5, verbose=1)

    #checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True, save_best_only=True, mode='auto', period=1)
    #tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=32)
    #history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=num_epochs, verbose=1, callbacks=[checkpoint, tensor_board], validation_split=0.2)
    
    tb_hist = TensorBoard(log_dir= graph_path , histogram_freq=0, write_graph=True, write_images=True)
    # sys.quit
    csv_logger = CSVLogger(os.path.join(log_path, 'log.out'), append=True, separator=';')
    
    history = model.fit_generator(
        train_batches,
        validation_data = valid_batches,
        epochs = epochs,
        verbose = 1,
        callbacks=[
            early_stopping, 
            model_checkpoint, 
            reduce_lr,
            tb_hist,
            csv_logger
        ],
        use_multiprocessing=True,
        workers=8,
    )
    
    return history
        
        
def main():
    # args  = argument()
    # args = args.parse_args()
    
    df = df_gen(FLAGS.json_path)
    img_path = '/'.join(FLAGS.json_path.split('/')[:-2])
    print(df.shape)
    print(img_path)
    # sys.quit
    
    if FLAGS.augument:
        argument = augumentation()
    else:
        argument = None
    
    train_idx, valid_idx = train_split(df)
    print("Train : {}\n validataion : {}".format(len(train_idx), len(valid_idx)))
    
    
    train_batches = DataGenerator(df.iloc[train_idx], batch_size = FLAGS.batch_size, 
                                  subset='train', shuffle=True,
                                 preprocess = img_preprocess, augmentation = argument,
                                 resize_shape = FLAGS.resize_shape, train_path = img_path)
    
    valid_batches = DataGenerator(df.iloc[valid_idx], batch_size = FLAGS.batch_size, 
                                  subset='train', shuffle=False,
                                 preprocess = img_preprocess, augmentation = None,
                                 resize_shape = FLAGS.resize_shape, train_path = img_path)
    
    seg_model = get_model(label_counts = len(df.columns[2:].tolist()), input_shape = (FLAGS.resize_shape[0],FLAGS.resize_shape[1],3))
    
    
    
    ### path_name set
    
    img_dir_name = img_path.split('/')[-1]
    task_name = 'segment'
    model_name = FLAGS.model_name
    main_path = './res/{}/{}/{}/'.format(img_dir_name, task_name, model_name)
    
    model_path = os.path.join(main_path, 'model/')
    graph_path = os.path.join(main_path, 'graph/')
    log_path = os.path.join(main_path, 'log/')
    res_path = os.path.join(main_path, 'res/')
    
    for path in [model_path, graph_path, log_path, res_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    
    
    print(model_path)
    print(graph_path)
    print(log_path)
    print(res_path)
    
    history = compile_and_train(seg_model, train_batches, valid_batches, int(FLAGS.epoch),
                               model_path = model_path, graph_path = graph_path, log_path = log_path)
    

    
    
    print('end')
    return None

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'testing')
    parser.add_argument('--target', required=True, help = 'train or predict')
    # parser.add_argument('--train_path')
    # parser.add_argument('--test_path')
    parser.add_argument('--json_path', required=True)
    parser.add_argument('--model_name',default= local_time)
    parser.add_argument('--epoch', default = 50, type =int)
    parser.add_argument('--batch_size', default = 16, type =int)
    
    parser.add_argument('--resize_shape', default = (256,256))
    parser.add_argument('--augument',default = True)
    
    
    FLAGS, unparsed = parser.parse_known_args()
    
    
    main()
    
    
    
    