from flask import Flask, request, send_file
import io
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from PIL import Image
import autokeras as ak
import tarfile
import zipfile
import json

app = Flask(__name__)
model = None


# Endpoint to receive blob data
@app.route('/predict', methods=['POST'])
def predict_blob():
    model = load_model('extracted_contents/Chicken Model', custom_objects=ak.CUSTOM_OBJECTS)


    with open('label_names.json', 'r') as f:
        label_names = json.load(f)


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

    print(prediction)

    label_names_swapped = {v: k for k, v in label_names.items()}
    predictions_array = []

    if(len(prediction) == 1):
        predictions_array.append([label_names_swapped[0], prediction[0][0]])
        predictions_array.append([label_names_swapped[1], 1-prediction[0][0]])

    else:
        label_index = 0
        for x in label_names:
            predictions_array.append([label_names_swapped[x], prediction[0][label_index]])
            label_index += 1

    print(predictions_array)

    # Convert the prediction to bytes
    predictions_json = json.dumps([[item[0], float(item[1])] for item in predictions_array])
    predictions_bytes = predictions_json.encode('utf-8')

    # Return the combined data as a single response
    return send_file(io.BytesIO(predictions_bytes), mimetype='application/octet-stream')



def extract_zip(zip_filename, extract_to):
    # Open the zip file
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        # Extract all the contents to the specified directory
        zip_ref.extractall(extract_to)



# Load model from desktop

def load_model_from_desktop(model_url):
    global model
    global label_names
    # Specify the name of the zip file
    zip_file_name = 'Chicken_model_directory.zip'

    # Specify the directory to extract the contents to
    extract_to_directory = 'extracted_contents'

    # Call the function to extract the zip file
    extract_zip(zip_file_name, extract_to_directory)
    model = load_model('extracted_contents/Chicken Model', custom_objects=ak.CUSTOM_OBJECTS)


    with open('label_names.json', 'r') as f:
        label_names = json.load(f)

    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000, debug=True)
    
