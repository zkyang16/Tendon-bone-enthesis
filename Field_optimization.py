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
import openpyxl
import pandas as pd


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
    
    # Output layer
    outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    
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
        return img / 255.0  

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

m = 11   
n = 11

m_max = 13  
m_min = 1
a_max = 75    
a_min = 5  
o_x_max = 1    
o_x_min = -1
o_y_max = 1  
o_y_min = 0
o_z_max = 1  
o_z_min = -1


k_scale = 2.5
ini_input = np.random.uniform(-1, 1, (5*m-8, n))
ini_input = ini_input - ini_input


def x_to_column(x):
    c = 512 + x*886/200.0
    
    return round(c)

def column_to_x(c):
    x = (c-512)*200.0/886.0
    
    return x

def y_to_row(y):
    r = 463 - y*443/100.0
    
    return round(r)

def row_to_y(r):
    y = (463-r)*100.0/443.0
    
    return y


def define_xi():
    xi = np.zeros((11,11))
    for i in range(11):
        xi[i,0] = 125
        xi[i,1] = 105
        xi[i,2] = 85
        xi[i,3] = 65
        xi[i,4] = 45
        xi[i,5] = 25
        xi[i,6] = 5
        xi[i,7] = -15
        xi[i,8] = -35
        xi[i,9] = -55
        xi[i,10] = -75
        
    return xi

def define_yi():
    yi = np.zeros((11,11))
    for j in range(11):
        yi[0,j] = 125
        yi[1,j] = 105
        yi[2,j] = 85
        yi[3,j] = 65
        yi[4,j] = 45
        yi[5,j] = 25
        yi[6,j] = 5
        yi[7,j] = -15
        yi[8,j] = -35
        yi[9,j] = -55
        yi[10,j] = -75
        
    return yi
        

def create_kernel(m,n):
    k = np.zeros((m,n,704, 1024), dtype=np.float32)
    sigma = 18.0
    for i in range(m):
        for j in range(n):
            print('i=',i,'j=',j)
            for y in range(704):
                for x in range(512,1024):
                    x0 = column_to_x(x)
                    y0 = row_to_y(y)
                    k[i,j,y,x] = np.exp(-((x0-xi[i,j])**2+(y0-yi[i,j])**2)/(2*sigma**2))
    
    for i in range(m):
        for j in range(n):
            print('i=',i,'j=',j)
            for y in range(704):
                for x in range(512):
                    k[i,j,y,x] = k[i,j,y,1023-x]
    
    return tf.convert_to_tensor(k, dtype=tf.float32)

def create_kernel2(m,n):
    k = np.zeros((m,n,704, 1024), dtype=np.float32)
    sigma = 10.0
    for i in range(m):
        for j in range(n):
            print('i=',i,'j=',j)
            for y in range(704):
                for x in range(512,1024):
                    x0 = column_to_x(x)
                    y0 = row_to_y(y)
                    xi = column_to_x(700)
                    yi = row_to_y(450)
                    k[i,j,y,x] = np.exp(-((x0-xi)**2+(y0-yi)**2)/(2*sigma**2))
    
    for i in range(m):
        for j in range(n):
            print('i=',i,'j=',j)
            for y in range(704):
                for x in range(512):
                    k[i,j,y,x] = k[i,j,y,1023-x]
    
    return tf.convert_to_tensor(k, dtype=tf.float32)
    

def superposition(alpha,k):
    P = tf.zeros((704, 1024), dtype=tf.float32)
    for i in range(m):
        for j in range(n):
            P = P + alpha[i,j]*k[i,j,:,:]
    P = P*(0.02-0.98)/(2*k_scale)+(0.02+0.98)/2

    return P

def gradient(top, bottom):
    
    y_indices = tf.range(704, dtype=tf.float32)

    P = top + (bottom - top) * (y_indices[:, None] / 703.0)

    return P
    
    
def Map(optimized_input,k):
    
    optimized_input = tf.cast(optimized_input, dtype=tf.float32)
    local_min = np.zeros((11,11))
    local_min[1,2] = -.5
    local_min = tf.cast(local_min, dtype=tf.float32)
      
    alpha1 = optimized_input[0:m-4,:]
    alpha1_top1 = tf.zeros(shape=(1, 11), dtype=tf.float32) -0.5
    alpha1_top2 = tf.zeros(shape=(1, 11), dtype=tf.float32) -0.4
    alpha1_bottom1 = tf.zeros(shape=(1, 11), dtype=tf.float32) + 0.4
    alpha1_bottom2 = tf.zeros(shape=(1, 11), dtype=tf.float32) + 0.5
    alpha1 = tf.concat([alpha1_top2, alpha1, alpha1_bottom1], axis=0)
    alpha1 = tf.concat([alpha1_top1, alpha1, alpha1_bottom2], axis=0)
    input1 = superposition(alpha1,k)
    
    alpha2 = optimized_input[m-4:2*m-8,:]
    alpha2_top1 = tf.zeros(shape=(1, 11), dtype=tf.float32) -0.5
    alpha2_top2 = tf.zeros(shape=(1, 11), dtype=tf.float32) -0.4
    alpha2_bottom1 = tf.zeros(shape=(1, 11), dtype=tf.float32) + 0.4
    alpha2_bottom2 = tf.zeros(shape=(1, 11), dtype=tf.float32) + 0.5
    alpha2 = tf.concat([alpha2_top2, alpha2, alpha2_bottom1], axis=0)
    alpha2 = tf.concat([alpha2_top1, alpha2, alpha2_bottom2], axis=0)
    input2 = superposition(alpha2,k)
    
    input3 = superposition(optimized_input[2*m-8:3*m-8,:],k)
    input4 = superposition(optimized_input[3*m-8:4*m-8,:],k)
    input5 = superposition(optimized_input[4*m-8:5*m-8,:],k)

    input1 = input1 + mask[:, :, 0]
    input2 = input2 + mask[:, :, 0]
    input3 = input3 + mask[:, :, 0]
    input4 = input4 + mask[:, :, 0]
    input5 = input5 + mask[:, :, 0]

    input1 = tf.clip_by_value(input1, 0.02, 0.98)
    input2 = tf.clip_by_value(input2, 0.02, 0.98)
    input3 = tf.clip_by_value(input3, 0.02, 0.98)
    input4 = tf.clip_by_value(input4, 0.02, 0.98)
    input5 = tf.clip_by_value(input5, 0.02, 0.98)

    batch_input = tf.stack((input1, input2, input3, input4, input5), axis=2)
    batch_input = tf.expand_dims(batch_input, axis=0)  

    return batch_input
         
def Map_alpha(optimized_input,k):
    
    optimized_input = tf.cast(optimized_input, dtype=tf.float32)
    local_min = np.zeros((11,11))
    local_min[1,2] = -.5
    local_min = tf.cast(local_min, dtype=tf.float32)
    
    
    alpha1 = optimized_input[0:m-4,:]
    alpha1_top1 = tf.zeros(shape=(1, 11), dtype=tf.float32) -0.5
    alpha1_top2 = tf.zeros(shape=(1, 11), dtype=tf.float32) -0.4
    alpha1_bottom1 = tf.zeros(shape=(1, 11), dtype=tf.float32) + 0.4
    alpha1_bottom2 = tf.zeros(shape=(1, 11), dtype=tf.float32) + 0.5
    alpha1 = tf.concat([alpha1_top2, alpha1, alpha1_bottom1], axis=0)
    alpha1 = tf.concat([alpha1_top1, alpha1, alpha1_bottom2], axis=0)
    
    alpha2 = optimized_input[m-4:2*m-8,:]
    alpha2_top1 = tf.zeros(shape=(1, 11), dtype=tf.float32) -0.5
    alpha2_top2 = tf.zeros(shape=(1, 11), dtype=tf.float32) -0.4
    alpha2_bottom1 = tf.zeros(shape=(1, 11), dtype=tf.float32) + 0.4
    alpha2_bottom2 = tf.zeros(shape=(1, 11), dtype=tf.float32) + 0.5
    alpha2 = tf.concat([alpha2_top2, alpha2, alpha2_bottom1], axis=0)
    alpha2 = tf.concat([alpha2_top1, alpha2, alpha2_bottom2], axis=0)
    
    alpha3 = optimized_input[2*m-8:3*m-8,:]
    alpha4 = optimized_input[3*m-8:4*m-8,:]
    alpha5 = optimized_input[4*m-8:5*m-8,:]

    return alpha1, alpha2, alpha3, alpha4, alpha5
    
    
def inverse_prediction(trained_model, ini_input, target_output, mask, k, num_iterations=500, learning_rate=0.01):
    """
    Parameters:
    - trained_model: The trained CNNFP
    - target_output: The desired output field
    - input_shape: The shape of the model's input.
    - num_iterations: The number of optimization steps.
    - learning_rate: The learning rate for optimization.
    
    Returns:
    - optimized_input: The input that produces or approximates the target output.
    """
    # Create a random input tensor with the same shape as the input layer
    optimized_input = tf.Variable(ini_input, trainable=True)
    print(optimized_input.shape)
    
    
    # Define an optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Loss Tracking"
    sheet.append(["Iteration", "Loss"])
    
    
    # Optimization loop
    for iteration in range(num_iterations):
        with tf.GradientTape() as tape:
            tape.watch(optimized_input)  
            mapped_input = Map(optimized_input,k)  
            predicted_output = trained_model(mapped_input, training=False)
            loss = -tf.reduce_min(predicted_output)
        
        # Compute gradients
        gradients = tape.gradient(loss, [optimized_input])
        if gradients[0] is None:
            raise ValueError("Gradients are None.")
        # Update the optimized input
        optimizer.apply_gradients(zip(gradients, [optimized_input]))
        
        # Apply constraints to keep optimized_input within rational range
        optimized_input.assign(tf.clip_by_value(optimized_input, clip_value_min= -0.4, clip_value_max= 0.4))
        
        sheet.append([iteration, loss.numpy()])
        
        if iteration % 50 == 0:
            print(f"Iteration {iteration}, Loss: {loss.numpy()}")
    
    workbook.save("loss_tracking.xlsx")
    print("Loss values saved to loss_tracking.xlsx")
    
    return optimized_input.numpy()


train_generator = ImagePairGenerator(
    input_dir1='inputs1',
    input_dir2='inputs2',
    input_dir3='inputs3',
    input_dir4='inputs4',
    input_dir5='inputs5',
    output_dir='outputs',
    batch_size=1,
    in_img_size=(751,1497),
    out_img_size = (704,1024),
    shuffle=True  
)


# Get a new batch of inputs
input_batch, output_batch = train_generator.__getitem__(0) 

mask = np.zeros((704,1024,1))+5
for i in range(704):
    for j in range(1024):
        if input_batch[0,i,j,0]<0.96:    
            input_batch[0,i,j,0] = 0.7 - i/704*0.4   # mineralization, not in effect
            input_batch[0,i,j,1] = 0.7 - i/704*0.4   # angular dispersion, not in effect
            input_batch[0,i,j,2] = 0.5   # mean orientation, not in effect
            output_batch[0,i,j,0] = 0.9
            mask[i,j,0] = 0
            

                
model  = load_model('CNNFP.keras')
model.summary()

xi = define_xi()
yi = define_yi()
k = create_kernel(m,n)
k2 = create_kernel2(m,n)

# Define a target output
target_output = output_batch[0,:,:,:]

# Perform inverse prediction
optimized_input = inverse_prediction(model, ini_input, target_output, mask, k, num_iterations=300, learning_rate=0.05)
print(optimized_input)
resulted_output = model(Map(optimized_input,k), training=False)
resulted_output = resulted_output[0,:,:,:]

df = pd.DataFrame(optimized_input)
df.to_excel("optimized_input.xlsx", index=False, header=False)

opt = Map_alpha(optimized_input,k) 
for i in range(5):
    array = opt[i].numpy()
    np.savetxt(f'alpha_{i+1}.txt', array, fmt='%.3f', delimiter=' ')

initial_output = model(Map(tf.Variable(ini_input, trainable=True),k), training=False)
initial_output = initial_output[0,:,:,:]


batch_input = Map(optimized_input,k)
for i in range(5):
    mt = batch_input[0,:,:,i:i+1]
    mt = np.concatenate((mt, mt, mt), axis=2)
    plt.imshow(mt)
    plt.savefig('optimal_input_'+str(i+1), bbox_inches='tight', pad_inches=0)
    
    
    
batch_input = Map(tf.Variable(ini_input, trainable=True),k)
for i in range(5):
    mt = batch_input[0,:,:,i:i+1]
    mt = np.concatenate((mt, mt, mt), axis=2)
    plt.imshow(mt)
    plt.savefig('initial_input_'+str(i+1), bbox_inches='tight', pad_inches=0)

    

m = target_output
m = np.concatenate((m, m, m), axis=2)
plt.imshow(m)
plt.savefig('target_output', bbox_inches='tight', pad_inches=0)

m = resulted_output
m = np.concatenate((m, m, m), axis=2)
plt.imshow(m)
plt.savefig('resulted_output', bbox_inches='tight', pad_inches=0)

m = mask
m = np.concatenate((m, m, m), axis=2)
plt.imshow(m)
plt.savefig('a_mask', bbox_inches='tight', pad_inches=0)

m = initial_output
m = np.concatenate((m, m, m), axis=2)
plt.imshow(m)
plt.savefig('initial_output', bbox_inches='tight', pad_inches=0)






