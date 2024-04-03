from flask import Flask, request, send_file
import io
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from PIL import Image
import autokeras as ak
import tarfile

app = Flask(__name__)
model = None


# Endpoint to receive blob data
@app.route('/predict', methods=['POST'])
def predict_blob():
    blob_data = request.data
    image_stream = io.BytesIO(blob_data)
    
    # Open the image using PIL
    img = Image.open(image_stream)
    
    # Resize the image to 256x256
    img = img.resize((256, 256))

    # Convert the image to an array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the pixel values


    # Perform predictions
    prediction = model.predict(img_array)


    # Convert the prediction to bytes
    prediction_bytes = prediction.tobytes()

    # Convert the preprocessed image to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Concatenate the prediction and image bytes
    combined_data = prediction_bytes + img_byte_arr

    # Return the combined data as a single response
    return send_file(io.BytesIO(combined_data), mimetype='application/octet-stream')


# Load model from desktop

def load_model_from_desktop(model_url):
    global model
    model = load_model(model_url, custom_objects=ak.CUSTOM_OBJECTS)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000, debug=True)
    
