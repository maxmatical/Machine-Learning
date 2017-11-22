library(keras)


# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y


# Input image dimensions
img_rows <- 28
img_cols <- 28

# Redefine  dimension of train/test inputs
dim(x_train) = c(nrow(x_train), img_rows, img_cols, 1) # c(n, x, y, 1)
dim(x_test) = c(nrow(x_test), img_rows, img_cols, 1)
dim(x_train)

#x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1)) 
#x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

# use data augmentation for images (COMES AFTER PREPROCESSING)
datagen = image_data_generator(
  rotation_range=40,
  width_shift_range=0.2,
  height_shift_range=0.2,
  rescale=1./255,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip= TRUE,
  vertical_flip = TRUE,
  fill_mode = "nearest"
)
datagen %>% fit_image_data_generator(x_train)
flow_images_from_data(x_train, y_train, generator = datagen,
                      batch_size = 32
                      )

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)


# Early Stopping callback - stops at patience+1 epoch after min(val_loss) doesn't decrease
EarlyStopping = callback_early_stopping(monitor = "val_loss", min_delta = 0, patience = 2,
                                        verbose = 0, mode = c("auto", "min", "max"))

# define input shape
shape = c(img_rows, img_cols, 1)

# create model
model = keras_model_sequential() 
model %>% 
  layer_conv_2d(filter = 64, kernel_size = c(3,3), activation = 'relu', input_shape = shape) %>%
  layer_zero_padding_2d(padding = c(1,1)) %>%
  layer_conv_2d(filter = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.5) %>% 
  
  layer_zero_padding_2d(padding = c(1,1)) %>%
  layer_conv_2d(filter = 128, kernel_size = c(3,3), activation = 'relu') %>%
  layer_zero_padding_2d(padding = c(1,1)) %>%
  layer_conv_2d(filter = 128, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.5) %>% 
  
  layer_flatten() %>% 
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 10, activation = 'softmax')
  
# compile
opt = optimizer_rmsprop(lr = 0.0001, decay = 1e-6)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = opt,
  metrics = c('accuracy')
)

# train
history = model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  callbacks = EarlyStopping,
  validation_split = 0.2
)


# using augmented data
model %>% fit_generator(
  flow_images_from_data(x_train, y_train, datagen, batch_size = 32),
  steps_per_epoch = as.integer(5000/32), 
  epochs = 30, 
  validation_data = list(x_test, y_test)
)
