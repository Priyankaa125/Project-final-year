import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Path to a sample image
sample_image_path = 'path/to/sample/image.jpg'

# Load and preprocess the image
image = load_img(sample_image_path, target_size=(48, 48))
image_array = img_to_array(image)

# Display the image
plt.imshow(image_array.astype('uint8'))
plt.axis('off')  # Hide axes
plt.show()
