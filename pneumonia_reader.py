import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
#from google.colab import files
from keras.preprocessing import image
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

normal_dir=os.path.join('D:/Python/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL')
pneumo_dir=os.path.join('D:/Python/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA')

train_n_names = os.listdir(normal_dir)
train_p_names = os.listdir(pneumo_dir)

print('total training normal images:', len(os.listdir(normal_dir)))
print('total training pneumonia images:', len(os.listdir(pneumo_dir)))

##matplotlib.pyplot.ion()/matplotlib.pyplot.ioff()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_n_pix = [os.path.join(normal_dir, fname) 
                for fname in train_n_names[pic_index-8:pic_index]]
next_p_pix = [os.path.join(pneumo_dir, fname) 
                for fname in train_p_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_n_pix+next_p_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()


#our CNN
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('normal') and 1 for the other ('pneumonia')
    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.summary()

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        'D:/Python/chest-xray-pneumonia/chest_xray/chest_xray/train/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1)

# uploaded = files.upload()

# for fn in uploaded.keys():

#   # predicting images
#   path = '/content/' + fn
#   img = image.load_img(path, target_size=(300, 300))
#   x = image.img_to_array(img)
#   x = np.expand_dims(x, axis=0)

#   images = np.vstack([x])
#   classes = model.predict(images, batch_size=10)
#   print(classes[0])
#   if classes[0]>0.5:
#     print(fn + " shows pneumonia")
#   else:
#     print(fn + " is normal")

##successive_outputs = [layer.output for layer in model.layers[1:]]
###visualization_model = Model(img_input, successive_outputs)
##visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
### Let's prepare a random input image from the training set.
##horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
##human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
##img_path = random.choice(horse_img_files + human_img_files)
##
##img = load_img(img_path, target_size=(300, 300))  # this is a PIL image
##x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
##x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)
##
### Rescale by 1/255
##x /= 255
##
### Let's run our image through our network, thus obtaining all
### intermediate representations for this image.
##successive_feature_maps = visualization_model.predict(x)
##
### These are the names of the layers, so can have them as part of our plot
##layer_names = [layer.name for layer in model.layers]
##
### Now let's display our representations
# for layer_name, feature_map in zip(layer_names, successive_feature_maps):
#   if len(feature_map.shape) == 4:
#     # Just do this for the conv / maxpool layers, not the fully-connected layers
#     n_features = feature_map.shape[-1]  # number of features in feature map
#     # The feature map has shape (1, size, size, n_features)
#     size = feature_map.shape[1]
#     # We will tile our images in this matrix
#     display_grid = np.zeros((size, size * n_features))
#     for i in range(n_features):
#       # Postprocess the feature to make it visually palatable
#       x = feature_map[0, :, :, i]
#       x -= x.mean()
#       x /= x.std()
#       x *= 64
#       x += 128
#       x = np.clip(x, 0, 255).astype('uint8')
#       # We'll tile each filter into this big horizontal grid
#       display_grid[:, i * size : (i + 1) * size] = x
#     # Display the grid
#     scale = 20. / n_features
#     plt.figure(figsize=(scale * n_features, scale))
#     plt.title(layer_name)
#     plt.grid(False)
#     plt.imshow(display_grid, aspect='auto', cmap='viridis')
