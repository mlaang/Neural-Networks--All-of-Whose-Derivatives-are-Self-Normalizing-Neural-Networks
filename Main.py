import numpy as np
import tensorflow as tf
import cv2

import Builders

#It is critical that the input image is square due to later calculations.
input_image = cv2.imread('cameraman_test_image.png', cv2.IMREAD_GRAYSCALE)
input_image = input_image/255.0
input_image = input_image.astype('float32')

def get_mgrid(sidelen, dim=2):
    ls = tf.linspace(-1.0, 1.0, num=sidelen)
    mg = tf.meshgrid(ls,ls)
    s = tf.stack(mg,axis=2)
    return tf.reshape(s, [-1,2])
    
shape = [2, 252, 256, 128, 128, 128, 128, 1]

F = Builders.build_orthogonal_model_with_rotated_encoder(shape, use_bias=True, scale_factor=0.5)

reshaped_image = tf.reshape(input_image, [input_image.shape[0]*input_image.shape[0], 1])
c = get_mgrid(input_image.shape[0])

loss_fn = tf.keras.losses.MeanSquaredError()
opt_aggressive = tf.keras.optimizers.Adam(learning_rate=0.001)
opt_less_aggressive = tf.keras.optimizers.Adam(learning_rate=0.0002)
reduce_lr_aggressive = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5)
reduce_lr_slower = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=40)
logger = tf.keras.callbacks.CSVLogger('training.log')

class WriteVideoFrameCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.videowriter = cv2.VideoWriter('frames.avi',cv2.VideoWriter_fourcc('X','V','I','D'),20.0,(input_image.shape[0],input_image.shape[0]),isColor=False)
        
    def __del__(self):
        self.videowriter.release()

    def on_epoch_end(self, epochs, logs=None):
        self.videowriter.write(np.array(255.0*tf.clip_by_value(tf.reshape(F(get_mgrid(input_image.shape[0])), [input_image.shape[0], input_image.shape[0]]), 0.0, 1.0),dtype=np.int8))

#Train aggressive is able to bring build_orthogonal_model_with_rotated_encoder([2, 252, 256, 128, 128, 128, 128, 1], use_bias=False, scale_factor=0.5) to a PSNR around 63.5
#in around 100 iterations.
def train_aggressive(iters):
    F.compile(optimizer=opt_aggressive, loss=loss_fn)
    F.fit(get_mgrid(input_image.shape[0]),reshaped_image, callbacks=[reduce_lr_aggressive, logger, WriteVideoFrameCallback()], epochs=iters)
    
#Train aggressive is able to bring build_orthogonal_model_with_rotated_encoder([2, 252, 256, 128, 128, 128, 128, 1], use_bias=False, scale_factor=0.5) to a PSNR around 63.5
#in around 100 iterations.
def train_less_aggressive(iters):
    F.compile(optimizer=opt_less_aggressive, loss=loss_fn)
    F.fit(get_mgrid(input_image.shape[0]),reshaped_image, callbacks=[reduce_lr_slower, logger, WriteVideoFrameCallback()], epochs=iters)

train_less_aggressive(400)