import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten, Conv2DTranspose
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model


def create_cnn_model(input_shape=(704,1024,5)):
    inputs = Input(input_shape)   # Input layer expects a 5-channel

    x = Conv2D(32, (3, 3), padding='same')(inputs) 
    x = Activation('relu')(x)  
    x = MaxPooling2D((2, 2), padding='same')(x) 
    
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)  
    x = MaxPooling2D((2, 2), padding='same')(x)  
    
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)  
    x = MaxPooling2D((2, 2), padding='same')(x)  
    
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)  
    x = MaxPooling2D((2, 2), padding='same')(x)  
    
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)  
    x = MaxPooling2D((2, 2), padding='same')(x)  
    
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)  
    x = MaxPooling2D((2, 2), padding='same')(x) 
    
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)  

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)  
    
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)  
    
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)  
    
    # Decoder starts
    
    x = Conv2DTranspose(128, (3, 3), padding='same', strides=(2, 2))(x)
    x = Activation('relu')(x)  
    
    x = Conv2DTranspose(128, (3, 3), padding='same', strides=(2, 2))(x)
    x = Activation('relu')(x)  

    x = Conv2DTranspose(128, (3, 3), padding='same', strides=(2, 2))(x)
    x = Activation('relu')(x)  

    x = Conv2DTranspose(128, (3, 3), padding='same', strides=(2, 2))(x)
    x = Activation('relu')(x)  
    
    x = Conv2DTranspose(64, (3, 3), padding='same', strides=(2, 2))(x)
    x = Activation('relu')(x)  
    
    x = Conv2DTranspose(32, (3, 3), padding='same', strides=(2, 2))(x)
    x = Activation('relu')(x)  
    
    # Output layer: 1-channel output image
    outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Create the model
    model = Model(inputs, outputs)
    return model

class ImagePairGenerator(Sequence):
    def __init__(self, input_dir1, input_dir2, input_dir3, input_dir4, input_dir5, output_dir, batch_size=32, in_img_size=(150, 150),
                 out_img_size=(150,150), shuffle=False):
        self.input_dir1 = input_dir1
        self.input_dir2 = input_dir2
        self.input_dir3 = input_dir3
        self.input_dir4 = input_dir4
        self.input_dir5 = input_dir5
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.in_img_size = in_img_size
        self.out_img_size = out_img_size
        self.shuffle = shuffle
        
        self.input_images1 = [f for f in os.listdir(input_dir1) if os.path.isfile(os.path.join(input_dir1, f))]
        self.input_images2 = [f for f in os.listdir(input_dir2) if os.path.isfile(os.path.join(input_dir2, f))]
        self.input_images3 = [f for f in os.listdir(input_dir3) if os.path.isfile(os.path.join(input_dir3, f))]
        self.input_images4 = [f for f in os.listdir(input_dir4) if os.path.isfile(os.path.join(input_dir4, f))]
        self.input_images5 = [f for f in os.listdir(input_dir5) if os.path.isfile(os.path.join(input_dir5, f))]
        
        self.output_images = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
        self.indexes = np.arange(len(self.input_images1)) 
        
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.input_images1) / self.batch_size))  

    def load_image(self, path):
        img = load_img(path, target_size=self.in_img_size)
        img = img_to_array(img)
        return img / 255.0  # Normalize to [0, 1]

    def tri_img_matrix(self,i):
        L = round((self.in_img_size[1]-self.out_img_size[1])/2.0)
        R = self.in_img_size[1]-self.out_img_size[1]-L
        U = round((self.in_img_size[0]-self.out_img_size[0])/2.0)
        D = self.in_img_size[0]-self.out_img_size[0]-U
        
        m1 = self.load_image(os.path.join(self.input_dir1, self.input_images1[i]))
        m1 = m1[:,:,0:1]
        m2 = self.load_image(os.path.join(self.input_dir2, self.input_images2[i]))
        m2 = m2[:,:,0:1]
        m3 = self.load_image(os.path.join(self.input_dir3, self.input_images3[i]))
        m3 = m3[:,:,0:1]
        m4 = self.load_image(os.path.join(self.input_dir4, self.input_images4[i]))
        m4 = m4[:,:,0:1]
        m5 = self.load_image(os.path.join(self.input_dir5, self.input_images5[i]))
        m5 = m5[:,:,0:1]
        m = np.concatenate((m1, m2, m3, m4, m5), axis=2)
        m = m[U:-D,L:-R,:]
        return m
    
    def one_img_matrix(self,i):
        L = round((self.in_img_size[1]-self.out_img_size[1])/2.0)
        R = self.in_img_size[1]-self.out_img_size[1]-L
        U = round((self.in_img_size[0]-self.out_img_size[0])/2.0)
        D = self.in_img_size[0]-self.out_img_size[0]-U
        
        m = self.load_image(os.path.join(self.output_dir, self.output_images[i]))
        m = m[:,:,0:1]
        m = m[U:-D,L:-R,:]
        return m


    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Prepare input and output batches
        input_batch = np.array([
            self.tri_img_matrix(i) for i in batch_indexes 
        ])

        output_batch = np.array([
            self.one_img_matrix(i)
            for i in batch_indexes
        ])
  

        return input_batch, output_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


model = create_cnn_model(input_shape=(704,1024,5))
model.summary()
model.compile(optimizer=Adam(), loss='mse')


train_generator = ImagePairGenerator(
    input_dir1='inputs1',
    input_dir2='inputs2',
    input_dir3='inputs3',
    input_dir4='inputs4',
    input_dir5='inputs5',
    output_dir='outputs',
    batch_size=20,
    in_img_size=(751,1497),
    out_img_size = (704,1024),
    shuffle=True 
)

validation_generator = ImagePairGenerator(
    input_dir1='validation/inputs1',
    input_dir2='validation/inputs2',
    input_dir3='validation/inputs3',
    input_dir4='validation/inputs4',
    input_dir5='validation/inputs5',
    output_dir='validation/outputs',
    batch_size=20,
    in_img_size=(751,1497),
    out_img_size = (704,1024),
    shuffle=True  
)


checkpoint = ModelCheckpoint(
    filepath='CNNFP_{epoch:02d}.keras',
    save_freq=200,                         
    save_weights_only=False,              
    verbose=1,                            
    save_best_only=False,                 
    mode='auto'                           
)

csv_logger = CSVLogger('training_log.csv', append=False)

callbacks = [checkpoint, csv_logger]

# Train the model
history = model.fit(
    train_generator,  
    epochs=500,        
    validation_data=validation_generator,
    callbacks=callbacks,  
    steps_per_epoch=len(train_generator), 
    verbose = 1
)

