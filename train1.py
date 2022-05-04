from audioop import avg
import pandas as pd
import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

np.random.seed(143)

train_images_directory = '/srv/beegfs02/scratch/aegis_guardian/data/speed_prediction/speedChallenge/aegis_data/IMG'
train_csv_file_directory = '/srv/beegfs02/scratch/aegis_guardian/data/speed_prediction/speedChallenge/aegis_data/train_data.csv'

df = pd.read_csv('/srv/beegfs02/scratch/aegis_guardian/data/datasets/finetuning/training/Hammond/speed_pred/commaai-speed-challenge-master/train50k_processed.csv', header=None)
df.columns = ['image_path', 'time', 'speed']
print("Data Loaded Successfully...")
print(f"Dataset Size: {len(df)}")
print(df.head())

# video_fps = 12
# times = np.asarray(df['time'], dtype = np.float32) / video_fps
# speeds = np.asarray(df['speed'], dtype=np.float32)
# plt.plot(times, speeds, 'r-')
# plt.title('Speed vs Time')
# plt.xlabel('time (secs)')
# plt.ylabel('speed (mph)')
# plt.savefig('Ground_Truth_Speed_vs_Time.jpg')

def batch_shuffle(dframe):
    """
    Randomly shuffle pairs of rows in the dataframe, separates train and validation data
    generates a uniform random variable 0->9, gives 20% chance to append to valid data, otherwise train_data
    return tuple (train_data, valid_data) dataframes
    """
    randomized_list = np.arange(len(dframe)-1)
    np.random.shuffle(randomized_list)
    
    train_data = pd.DataFrame()
    valid_data = pd.DataFrame()
    test_data = pd.DataFrame()
    
    for i in randomized_list:
        idx1 = i
        idx2 = i + 1
        
        row1 = dframe.iloc[[idx1]].reset_index()
        row2 = dframe.iloc[[idx2]].reset_index()
        
        randInt = np.random.randint(10)
        if 0 <= randInt <= 1:
            valid_frames = [valid_data, row1, row2]
            valid_data = pd.concat(valid_frames, axis = 0, join = 'outer', ignore_index=False)
        if randInt == 2:
            test_frames = [test_data, row1, row2]
            test_data = pd.concat(test_frames, axis = 0, join = 'outer', ignore_index=False)
        if randInt > 2:
            train_frames = [train_data, row1, row2]
            train_data = pd.concat(train_frames, axis = 0, join = 'outer', ignore_index=False)
    return train_data, valid_data, test_data


# create training and validation set
train_data, valid_data, test_data = batch_shuffle(df)

# save to csv
train_data.to_csv('train.csv')
valid_data.to_csv('valid_data.csv')
test_data.to_csv('test_data.csv')

# verify data size
print('Training data size =', train_data.shape)
print('Validation data size =', valid_data.shape)
print('Test data size =', test_data.shape)



def change_brightness(image, bright_factor):
    """
    Augments the brightness of the image by multiplying the saturation by a uniform random variable
    Input: image (RGB)
    returns: image with brightness augmentation
    """
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # perform brightness augmentation only on the second channel
    hsv_image[:,:,2] = hsv_image[:,:,2] * bright_factor
    
    # change back to RGB
    image_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return image_rgb



def opticalFlowDense(image_current, image_next):
    prvs = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    hsv = np.zeros_like(image_current)
    hsv[...,1] = 255
    next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    return rgb

def crop_image(image, scale):
    """
    preprocesses the image
    
    input: image (480 (y), 640 (x), 3) RGB
    output: image (shape is (66, 220, 3) as RGB)
    
    This stuff is performed on my validation data and my training data
    Process: 
             1) Cropping out black spots
             3) resize to (66, 220, 3) if not done so already from perspective transform
    """
    # Crop out sky (top 130px) and the hood of the car (bottom 270px) 
    image_cropped = image[130:370,:] # -> (240, 640, 3)
    
    height = int(240*scale)
    width = int(640*scale)
    image = cv2.resize(image_cropped, (220, 66), interpolation = cv2.INTER_AREA)
    
    return image


def preprocess_image_valid_from_path(image_path, scale_factor=0.5):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image(img, scale_factor)
    return img

def preprocess_image_from_path(image_path, scale_factor=0.5, bright_factor=1):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = change_brightness(img, bright_factor)
    img = crop_image(img, scale_factor)
    return img


def generate_training_data(data, batch_size = 16, scale_factor = 0.5):
    # sample an image from the data to compute image size
    img = preprocess_image_from_path(train_data.iloc[1]['image_path'],scale_factor)

    # create empty batches
    image_batch = np.zeros((batch_size, img.shape[0], img.shape[1], img.shape[2]))
    label_batch = np.zeros(batch_size)
    i=0
    
    while True:
        speed1 = data.iloc[i]['speed']
        speed2 = data.iloc[i+1]['speed']
        
        bright_factor = 0.2 + np.random.uniform()
        img1 = preprocess_image_from_path(data.iloc[i]['image_path'],scale_factor,bright_factor)
        img2 = preprocess_image_from_path(data.iloc[i+1]['image_path'],scale_factor,bright_factor)
        
    
        rgb_flow_diff = opticalFlowDense(img1,img2)
        avg_speed = np.mean([speed1,speed2])
        
        image_batch[(i//2)%batch_size] = rgb_flow_diff
        label_batch[(i//2)%batch_size] = avg_speed
        
        if not(((i//2)+1)%batch_size):
            yield image_batch, label_batch
        i+=2
        i=i%data.shape[0]

def generate_validation_data(data, batch_size = 16, scale_factor = 0.5):
    # sample an image from the data to compute image size
    img = preprocess_image_from_path(train_data.iloc[1]['image_path'],scale_factor)

    # create empty batches
    image_batch = np.zeros((batch_size, img.shape[0], img.shape[1], img.shape[2]))
    label_batch = np.zeros(batch_size)
    i=0
    
    while True:
        speed1 = data.iloc[i]['speed']
        speed2 = data.iloc[i+1]['speed']
        
        bright_factor = 0.2 + np.random.uniform()
        img1 = preprocess_image_from_path(data.iloc[i]['image_path'],scale_factor,bright_factor)
        img2 = preprocess_image_from_path(data.iloc[i+1]['image_path'],scale_factor,bright_factor)
        
    
        rgb_flow_diff = opticalFlowDense(img1,img2)
        avg_speed = np.mean([speed1,speed2])
        
        image_batch[(i//2)%batch_size] = rgb_flow_diff
        label_batch[(i//2)%batch_size] = avg_speed

        i1 = data.iloc[i]['image_path']
        i2 = data.iloc[i+1]['image_path']
        # print(f'\nImage {i}: {i1}')
        # print(f'Image {i+1}: {i2}')
        # print(f'Image 1 processed: {np.average(img1)}')
        # print(f'Image 2 processed: {np.average(img2)}')
        # print(f'Optical flow avg: {np.average(rgb_flow_diff)}')
        # print(f'Label: {avg_speed}')
        
        if not(((i//2)+1)%batch_size):
            yield image_batch, label_batch
        i+=2
        i=i%data.shape[0]


from tensorflow import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers import ELU
# from tensorflow.keras.optimizers import Adam
import keras.backend as KTF


N_img_height = 66
N_img_width = 220
N_img_channels = 3
def nvidia_model():
    inputShape = (N_img_height, N_img_width, N_img_channels)

    model = Sequential()
    # normalization    
    # perform custom normalization before lambda layer in network
    model.add(Lambda(lambda x: x/ 127.5 - 1, input_shape = inputShape))

    model.add(Convolution2D(24, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv1'))
    
    
    model.add(ELU())    
    model.add(Convolution2D(36, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv2'))
    
    model.add(ELU())    
    model.add(Convolution2D(48, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv3'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, (3, 3), 
                            strides = (1,1), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv4'))
    
    model.add(ELU())              
    model.add(Convolution2D(64, (3, 3), 
                            strides= (1,1), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv5'))
              
              
    model.add(Flatten(name = 'flatten'))
    model.add(ELU())
    model.add(Dense(100, kernel_initializer = 'he_normal', name = 'fc1'))
    model.add(ELU())
    model.add(Dense(50, kernel_initializer = 'he_normal', name = 'fc2'))
    model.add(ELU())
    model.add(Dense(10, kernel_initializer = 'he_normal', name = 'fc3'))
    model.add(ELU())
    
    # do not put activation at the end because we want to exact output, not a class identifier
    model.add(Dense(1, name = 'output', kernel_initializer = 'he_normal'))
    
    adam = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer = adam, loss = 'mse')

    return model


# val_size = len(valid_data.index)
# train_size = len(train_data.index)
# valid_generator = generate_validation_data(valid_data)
# BATCH = 16

# from keras.callbacks import EarlyStopping, ModelCheckpoint

# filepath = 'deep_train_fixed3_2.h5'
# # earlyStopping = EarlyStopping(monitor='val_loss', 
# #                               patience=5, 
# #                               verbose=1, 
# #                               min_delta = 0.23,
# #                               mode='min')
# modelCheckpoint = ModelCheckpoint(filepath, 
#                                   monitor = 'val_loss', 
#                                   save_best_only = True, 
#                                   mode = 'min', 
#                                   verbose = 1,
#                                  save_weights_only = True)
# callbacks_list = [modelCheckpoint]

model = nvidia_model()
# model.load_weights('model-weights-Vtest3.h5')


# train_generator = generate_training_data(train_data, BATCH)
# history = model.fit(
#         train_generator, 
#         # steps_per_epoch = train_size/BATCH, 
#         steps_per_epoch = 400, 
#         epochs = 15,
#         callbacks = callbacks_list,
#         verbose = 1,
#         validation_data = valid_generator,
#         validation_steps = 400)

# print(history)
# print(history.history.keys())

### plot the training and validation loss for each epoch
# plt.figure(figsize=[10,8])
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model mean squared error loss per epoch')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.savefig('DeepTrainFixed2Loss.png')


# # TEST SCRIPT
print('\n\n\n\nPERFROMING TESTS NOW')
import time
start=start = time.time()
# load weight
model.load_weights('deep_train_fixed3_2.h5')


# test_data = pd.read_csv('/srv/beegfs02/scratch/aegis_guardian/data/Datasets_ML_Pipeline/Drive360Challenge/Drive360Images/drive360challenge_validation.csv')
# test_data = test_data[['cameraRight', 'canSpeed']]
# test_data.rename({'cameraRight': 'image_path', 'canSpeed': 'speed'}, axis=1, inplace=True)
# test_data.loc[:1000]


import os
predict_speed = []
labels = []
error = []

# i=0
scale_factor = 0.5
bright_factor=1
for i in range(0, len(test_data)-1):
    speed1 = test_data.iloc[i]['speed']
    speed2 = test_data.iloc[i+1]['speed']

    # path1 = '/srv/beegfs02/scratch/aegis_guardian/data/Datasets_ML_Pipeline/Drive360Challenge/Drive360Images/' + test_data.iloc[i]['image_path']
    # path2 = '/srv/beegfs02/scratch/aegis_guardian/data/Datasets_ML_Pipeline/Drive360Challenge/Drive360Images/' + test_data.iloc[i+1]['image_path']

    # print(path1)
    # print(path2)
    img1 = preprocess_image_from_path(test_data.iloc[i]['image_path'],scale_factor,bright_factor)
    img2 = preprocess_image_from_path(test_data.iloc[i+1]['image_path'],scale_factor,bright_factor)
    
    # for i in range(10):
    #         rgb_diff = opticalFlowDense(img1,img2)
    #         if (np.average(rgb_diff) != 0):
    #             break

    rgb_diff = opticalFlowDense(img1, img2)
    rgb_diff = rgb_diff.reshape(1, rgb_diff.shape[0], rgb_diff.shape[1], rgb_diff.shape[2])
    avg_speed = np.array([[np.mean([speed1,speed2])]])

    i1 = test_data.iloc[i]['image_path']
    i2 = test_data.iloc[i+1]['image_path']
    
    prediction = model.predict(rgb_diff)
    predict_speed.append(float(prediction[0][0]))
    err = abs(avg_speed[0][0]-prediction[0][0])
    labels.append(avg_speed[0][0])
    error.append(err)
    
    print('------------------------------------------------------------------------------------------')
    print(f'\nImage {i}: {i1}')
    print(f'Image {i+1}: {i2}')
    print(f'Image 1 processed: {np.average(img1)}')
    print(f'Image 2 processed: {np.average(img2)}')
    print(f'Optical flow avg: {np.average(rgb_diff)}')
    print(f'Truth: {avg_speed[0][0]} Prediction: {prediction[0][0]}  Error:{err}')
    print('------------------------------------------------------------------------------------------')
    i+=2

print('done!')


# plt.figure(1, figsize=(20,10))
# plt.plot(df['speed'], alpha=0.9, )
# plt.plot(predict_speed, alpha=0.5)
# plt.ylabel('speed (mps)')
# plt.title('Ground Truth vs Predictions')
# plt.legend(['ground truth', 'predictions'])
# plt.savefig('predictions_vs_groundTruth.jpg')

from sklearn.metrics import mean_squared_error
print(f"MSE: {mean_squared_error(labels, predict_speed)}")
# print(f'AVG of err array: {np.mean(error)}')

end  = time.time()
exc_time = end-start
print('********time************', round(exc_time, 5))

plt.figure(1, figsize=(20,10))
plt.plot(labels)
plt.plot(predict_speed)
plt.ylabel('speed (mph)')
plt.title('Ground Truth vs Predictions')
plt.legend(['ground truth', 'predictions'])
plt.savefig('predictions_vs_Truth2.jpg')