#Conv2D: This layer applies convolution operations on the input image.
#32: This specifies the number of filters (or kernels) to be applied. Each filter will learn to detect different features in the image.
#(3, 3): This is the size of each filter, indicating that it is a 3x3 pixel window that will slide over the input image.
#activation='relu': The Rectified Linear Unit (ReLU) activation function introduces non-linearity by transforming the output of the convolution to max(0, x). This helps the model learn complex patterns.
#input_shape=(48, 48, 1): This specifies the shape of the input images. Here, each input image is 48x48 pixels with a single channel (grayscale).