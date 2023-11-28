import pickle
import joblib
import tensorflow as tf
from PIL import Image
import numpy as np

model = joblib.load('digit_knn_model.sav')

converter = tf.lite.TFLiteConverter.from_saved_model(model)
tflite_model = converter.convert()


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="digit_svm_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_tensor_index = interpreter.get_input_details()[0]['index']
output = interpreter.tensor(interpreter.get_output_details()[0]['index'])

# Load the image
image_path = "path/to/your/folder/handwritten_image.png"
image = Image.open(image_path).convert("L")  # Convert to grayscale
image = image.resize((28, 28))  # Resize to match the MNIST input size

# Preprocess the image
input_data = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
input_data = np.expand_dims(input_data, axis=-1)

# Set the input tensor
interpreter.set_tensor(input_tensor_index, input_data)

# Run inference
interpreter.invoke()

# Get the output
output_data = interpreter.get_tensor(output()[0]['index'])

# Post-process the output to get the predicted class
predicted_class = np.argmax(output_data)

print(f"The predicted class is: {predicted_class}")
