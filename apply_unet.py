import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gc
import rasterio
from skimage import morphology
from skimage.filters import apply_hysteresis_threshold
from astropy.convolution import convolve

from keras import backend as K
from keras.callbacks import Callback
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, UpSampling2D, Input, concatenate, Activation
from skimage.filters import apply_hysteresis_threshold

def construct_parameter_dictionary(default=True,input_size=192,filters=32,num_layers=4,dropout=0.3,dropout_change=0.0):
    if default:
        return {'input_shape':(192, 192, 1),
                        'filters':32,
                        'use_batch_norm':True,
                        'dropout':0.3,
                        'dropout_change_per_layer':0.0,
                        'num_layers':4}
    else:
        return {'input_shape':(input_size, input_size, 1),
                        'filters':filters,
                        'use_batch_norm':True,
                        'dropout':dropout,
                        'dropout_change_per_layer':dropout_change,
                        'num_layers':num_layers}

def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

def conv2d_block(
    inputs,
    use_batch_norm=True, 
    dropout=0.3, 
    filters=16, 
    kernel_size=(3,3), 
    activation='relu', 
    kernel_initializer='he_normal', 
    padding='same'):
    
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding) (inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    
    if dropout > 0.0:
        c = Dropout(dropout)(c)
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding) (c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c

def custom_unet(
    input_shape,
    num_classes=1,
    use_batch_norm=True, 
    upsample_mode='deconv', # 'deconv' or 'simple' 
    use_dropout_on_upsampling=False, 
    dropout=0.3, 
    dropout_change_per_layer=0.0,
    filters=16,
    num_layers=4,
    output_activation='sigmoid'): # 'sigmoid' or 'softmax'
    

    if upsample_mode=='deconv':
        upsample=upsample_conv
    else:
        upsample=upsample_simple

    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs   

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)
        down_layers.append(x)
        x = MaxPooling2D((2, 2)) (x)
        dropout += dropout_change_per_layer
        filters = filters*2 # double the number of filters with each layer

    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):        
        filters //= 2 # decreasing number of filters with each layer 
        dropout -= dropout_change_per_layer
        x = upsample(filters, (2, 2), strides=(2, 2), padding='same') (x)
        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)
    
    x = Conv2D(num_classes, (1, 1)) (x)
    outputs = Activation(output_activation) (x)    
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def predict_image(LoG, model_filename, buffer=5, model_parameter_dict=construct_parameter_dictionary(),low_limit=None,high_limit=None,verbose=0,report_frequency=10):
    
    model = custom_unet(**model_parameter_dict)
    
    model.load_weights(model_filename)


    if low_limit is None:
        low_limit=np.min(LoG)

    if high_limit is None:
        high_limit=np.max(LoG)    
    
    LoG[LoG<low_limit] = low_limit
    LoG[LoG>high_limit] = high_limit
    
    # scale
    x = (LoG - low_limit) / (high_limit - low_limit)
    
    original_shape = LoG.shape

    input_shape=model_parameter_dict['input_shape']

    x_stride = input_shape[1] - (buffer * 2)
    
    xpad_extra = x_stride - ((original_shape[1] + (buffer * 2)) % x_stride) + (buffer * 2)
    
    y_stride = input_shape[0] - (buffer * 2)
    
    ypad_extra = y_stride - ((original_shape[0] + (buffer * 2)) % y_stride) + (buffer * 2)
    
    flt_arr = np.pad(x, [(buffer, buffer + ypad_extra),(buffer, buffer + xpad_extra)], mode='reflect')
    
    predicted_arr = np.zeros((original_shape[0] + ypad_extra, original_shape[1] + xpad_extra))
    
    completed = 0
    percentage = 0
    
    rows = int((flt_arr.shape[0] / y_stride))
    cols = int((flt_arr.shape[1] / x_stride))

    total = rows * cols
    
    print("Starting UNet application over LoG raster, a total of {} passes needed.".format(total))
    
    for row in range(1, rows + 1):
        row_int_0 = (row - 1) * y_stride
        row_int = row_int_0 + input_shape[0]
        
        out_row_int_0 = (row - 1) * y_stride
        out_row_int = row * y_stride
        
        for col in range(1, cols + 1):
            col_int_0 = (col - 1) * x_stride
            col_int = col_int_0  + input_shape[1]
            
            out_col_int_0 = (col - 1) * x_stride
            out_col_int = col * x_stride
            
            clip = flt_arr[row_int_0:row_int,col_int_0:col_int]
            clip = clip.reshape(1, input_shape[0], input_shape[1], 1)
            
            clip_pred = model.predict(clip,verbose=verbose)
                        
            predicted_arr[out_row_int_0:out_row_int,out_col_int_0:out_col_int] = clip_pred[0, buffer:-buffer, buffer:-buffer, 0]
            
            completed += 1
            
            percent_completed = round((completed / total) * 100, 0)
            
            if percent_completed % report_frequency == 0 and percent_completed > percentage:
                print("{}% tiles completed".format(percent_completed))
                percentage += report_frequency

    return predicted_arr[:-ypad_extra,:-xpad_extra]
