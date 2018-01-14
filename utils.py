# Background functions
import numpy as np
import matplotlib.pylab as plt
import os
import sys
import cv2
import h5py
import tensorflow as tf
import imgaug as ia # Image augmentation for machine learning
from imgaug import augmenters as iaa
import skvideo.io

def create_image(train_video_address,video_label_address,train_images_address,train_labels_address):
    video_labels= open(video_label_address).readlines()
    video=skvideo.io.vreader(train_video_address)

    count=0
    train_labels=open(train_labels_address,'w')
    
    for frame in video:
        print(count)
        frame_address=train_images_address+str(count)+'.png'
        cv2.imwrite(frame_address,cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        train_labels.write(video_labels[count])
        count+=1
    
    train_labels.close()
    print("Image frames and labels created.")
        
def display(raw_image):
    plt.imshow(raw_image[:,:,::-1].astype(np.uint8),cmap='gray')
    
sometimes=lambda aug:iaa.Sometimes(0.5,aug)
seq = iaa.Sequential([sometimes(iaa.GaussianBlur((0, 1.5)))],
                     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), 
                    iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2)]))

def augment_image(raw_images):
    augment_brightness=0.5*np.random.uniform()+0.5
    iaa_image=seq.augment_image(raw_images)
    hsv=cv2.cvtColor(iaa_image,cv2.COLOR_RGB2HSV)
    hsv[:,:,2]=hsv[:,:,2]*augment_brightness
    bright_image=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    
    return bright_image

class Data_BatchGenerator:
    def __init__(self,root_path,train_images_path,train_labels_path,indices,batch,timestep=1,augment=True,normalization=True):
        self.root=root_path
        self.indices=indices
        self.batch_size=batch
        self.timestep=timestep
        self.augment=augment
        self.normalization=normalization
        
        
        self.train_images=sorted(os.listdir(self.root+'images/'))
        self.train_labels=open(train_labels_path).readlines()
    
    def sizeof(self):
        temp=len(self.indices)/self.batch_size
        return temp

    def generator(self):
        H,W= 112,112
        timestep=20
        n_images=len(self.indices)
        
        lower_bound=0
        if self.batch_size<n_images:
            upper_bound=self.batch_size
        else:
            upper_bound=n_images

        shuffled_images_indices=np.random.shuffle(self.indices)
                           
        while True:
            if lower_bound==upper_bound:
                lower_bound=0
                upper_bound=self.batch_size if self.batch_size<n_images else n_images

                shuffled_images_indices=np.random.shuffle(self.indices)               

            xbatch=np.zeros((upper_bound-lower_bound,self.timestep,H,W,3))
            ybatch=np.zeros((upper_bound-lower_bound,1))
            current_index=0

            for ind in self.indices[lower_bound:upper_bound]:
                for i in range(self.timestep):
                    image_path=self.root+'images/'+self.train_images[ind-self.timestep+1+i]
                    
                    image=cv2.imread(image_path)
                    
                    height=image.shape[0]
                    image=image[int(height/3):int(height),:,:]
                    
                    image=cv2.resize(image.copy(),(H,W))

                    if self.augment==True:
                        image=augment_image(image)
                    if self.normalization==True:
                        image=image-[105,115,120]

                    xbatch[current_index,i]=image

                train_speed=[np.float(speed) for speed in self.train_labels[ind-self.timestep+1:ind+1]]
                ybatch[current_index]=np.mean(train_speed)
                current_index+=1
            yield xbatch,ybatch
            lower_bound=upper_bound
            upper_bound+=self.batch_size
            if upper_bound>n_images:
                upper_bound=n_images

def load_weights(keras_model,pretrained_weights):
    loaded_weights=h5py.File(pretrained_weights,mode='r')

    for layer_index in range(len(keras_model.layers)):
        layer=keras_model.layers[layer_index]
        weights=loaded_weights['layer_'+str(layer_index)].values()
        weights=[w.value for w in weights]
        weights=[w if len(w.shape)<4 else w.transpose(2,3,4,1,0) for w in weights]
        layer.set_weights(weights)

        if layer_index>(len(keras_model.layers)-3): # Retrain last two layers
            break
            
def squared_loss(y_true,y_pred):
    squared_loss=tf.reduce_mean(tf.squared_difference(y_true,y_pred))
    return squared_loss

if __name__=="__main__":
    train_video_address='dataset/data/train.mp4'
    video_label_address='dataset/data/train.txt'
    train_images_address='dataset/video_frames/images/'
    train_labels_address='dataset/video_frames/train_labels.txt'

    create_image(train_video_address,video_label_address,train_images_address,train_labels_address)