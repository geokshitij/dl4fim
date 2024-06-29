#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import yaml
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Dropout
from tensorflow.keras.models import Model, load_model
from sklearn.utils import class_weight
from geotile import GeoTile, mosaic, vectorize

# Load configuration
with open('config.yaml') as f:
    config = yaml.safe_load(f)

x_train_path = config['paths']['x_train']
y_train_path = config['paths']['y_train']
x_test_path = config['paths']['x_test']
y_test_path = config['paths']['y_test']
best_model_path = config['paths']['best_model']
prediction_tiles_path = config['paths']['prediction_tiles']
merged_raster_output = config['paths']['merged_raster_output']
vectorized_output = config['paths']['vectorized_output']
raster_input = config['paths']['raster_input']

# Load initial data
train_xx = np.load(x_train_path)
train_yy = np.load(y_train_path)

# Split data into training and test sets
train_xx_initial, test_xx, train_yy_initial, test_yy = train_test_split(train_xx, train_yy, test_size=0.2, random_state=42)
np.save(x_test_path, test_xx)
np.save(y_test_path, test_yy)
print(train_xx_initial.shape, train_yy_initial.shape, test_xx.shape, test_yy.shape)

# Data augmentation
seq = iaa.Sequential([
    iaa.Fliplr(0.5), 
    iaa.Flipud(0.5), 
    iaa.Affine(rotate=(-45, 45)), 
    iaa.Affine(scale=(0.5, 1.5)), 
    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
])

augmented_images, augmented_masks = [], []
num_augmentations = 3

for i in range(train_xx_initial.shape[0]):
    image, mask = train_xx_initial[i], train_yy_initial[i]
    segmentation_map = SegmentationMapsOnImage(mask, shape=image.shape)

    for _ in range(num_augmentations):
        augmented_image, augmented_segmentation_mask = seq.augment(image=image, segmentation_maps=segmentation_map)
        augmented_images.append(augmented_image)
        augmented_masks.append(augmented_segmentation_mask.get_arr())

augmented_images = np.array(augmented_images)
augmented_masks = np.array(augmented_masks)

# Append augmented data to original data
train_xx = np.concatenate((train_xx_initial, augmented_images), axis=0)
train_yy = np.concatenate((train_yy_initial, augmented_masks), axis=0)
del augmented_images, augmented_masks

print(train_xx.max(), train_xx.min(), train_xx.dtype)
print(train_xx.shape, train_yy.shape)

num_samples = train_xx.shape[0]
shuffled_indices = np.random.permutation(num_samples)
train_xx = train_xx[shuffled_indices]
train_yy = train_yy[shuffled_indices]

# Plot a sample input RGB image and output image with buildings
img = np.random.randint(0, 10)
plt.imshow(train_xx[img, :, :, :3])
plt.show()
plt.imshow(train_yy[img, :, :, 0])
plt.show()

# Calculate class weights
train_labels_flat = train_yy_initial.flatten()
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels_flat), y=train_labels_flat)
class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
print("Class weights tensor:", class_weights_tensor)

@tf.keras.utils.register_keras_serializable()
def custom_weighted_loss(class_weights_tensor):
    def loss(y_true, y_pred):
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        epsilon = tf.keras.backend.epsilon()
        y_pred_flat = tf.clip_by_value(y_pred_flat, epsilon, 1 - epsilon)
        loss = - (y_true_flat * tf.math.log(y_pred_flat) + (1 - y_true_flat) * tf.math.log(1 - y_pred_flat))
        weights = tf.gather(class_weights_tensor, tf.cast(y_true_flat, tf.int32))
        weighted_loss = loss * weights
        return tf.reduce_mean(weighted_loss)
    return loss

# Build the model
x_in = Input(shape=(256, 256, 14))

'''Encoder'''
x_temp = Conv2D(32, (3, 3), activation='relu', padding='same')(x_in)
x_temp = Dropout(0.25)(x_temp)
x_skip1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = MaxPooling2D((2,2))(x_skip1)
x_temp = Conv2D(32, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = Dropout(0.25)(x_temp)
x_skip2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = MaxPooling2D((2,2))(x_skip2)
x_temp = Conv2D(64, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = Dropout(0.25)(x_temp)
x_skip3 = Conv2D(64, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = MaxPooling2D((2,2))(x_skip3)
x_temp = Conv2D(64, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = Dropout(0.5)(x_temp)
x_temp = Conv2D(64, (3, 3), activation='relu', padding='same')(x_temp)

'''Decoder'''
x_temp = Conv2DTranspose(64, (3, 3), activation='relu',  padding='same')(x_temp)
x_temp = Dropout(0.5)(x_temp)
x_temp = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu',  padding='same')(x_temp)
x_temp = Concatenate()([x_temp, x_skip3])
x_temp = Conv2DTranspose(64, (3, 3), activation='relu',  padding='same')(x_temp)
x_temp = Dropout(0.5)(x_temp)
x_temp = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu',  padding='same')(x_temp)
x_temp = Concatenate()([x_temp, x_skip2])
x_temp = Conv2DTranspose(32, (3, 3), activation='relu',  padding='same')(x_temp)
x_temp = Dropout(0.5)(x_temp)
x_temp = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu',  padding='same')(x_temp)
x_temp = Concatenate()([x_temp, x_skip1])
x_temp = Conv2DTranspose(32, (3, 3), activation='relu',  padding='same')(x_temp)
x_temp = Dropout(0.5)(x_temp)
x_temp = Conv2DTranspose(32, (3, 3), activation='relu',  padding='same')(x_temp)

'''Use 1 by 1 Convolution to get desired output bands'''
x_temp = Conv2D(32, (1, 1), activation='relu', padding='same')(x_temp)
x_temp = Conv2D(32, (1, 1), activation='relu', padding='same')(x_temp)
x_out = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x_temp)

model = Model(inputs=x_in, outputs=x_out)
model.compile(loss=custom_weighted_loss(class_weights_tensor), optimizer='adam')
model.summary()

# Train the model
checkpointer = tf.keras.callbacks.ModelCheckpoint(best_model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
callbacks = [earlyStopping, checkpointer]

history = model.fit(train_xx, train_yy, validation_data=(test_xx, test_yy), epochs=3, batch_size=128, verbose=1, callbacks=callbacks)

# Plot model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Prediction
model = load_model(best_model_path)

threshold = 0.5
pred_test = model.predict(test_xx)
pred_test = (pred_test > threshold).astype(np.uint8)
print(pred_test.shape)

img = np.random.randint(0, 20)
plt.imshow(pred_test[img, :, :, 0])
plt.show()
plt.imshow(test_yy[img,:,:,0])
plt.show()

test_yy = test_yy.astype('uint8')
pred_test = pred_test.astype('uint8')

def calculate_metrics(ground_truth, predicted_mask):
    ground_truth = ground_truth.ravel()
    predicted_mask = predicted_mask.ravel()

    TP = np.sum((ground_truth == 1) & (predicted_mask == 1))
    FP = np.sum((ground_truth == 0) & (predicted_mask == 1))
    FN = np.sum((ground_truth == 1) & (predicted_mask == 0))
    TN = np.sum((ground_truth == 0) & (predicted_mask == 0))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    iou = TP / (TP + FP + FN)

    return precision, recall, f1_score, iou

print(calculate_metrics(test_yy, pred_test))

# GeoTile operations
gt = GeoTile(raster_input)
gt.meta

gt.generate_tiles(save_tiles=False, stride_x=256, stride_y=256)
gt.convert_nan_to_zero()
gt.normalize_tiles()

threshold = 0.5
pred_test = model.predict(gt.tile_data)
pred_test = (pred_test > threshold).astype(np.uint8)
print(pred_test.shape)

gt.tile_data = pred_test
gt.save_tiles(prediction_tiles_path)

mosaic(prediction_tiles_path, merged_raster_output)
vectorize(merged_raster_output, vectorized_output, raster_values=[1])
