import logging
import numpy as np
import cv2
import json
import azure.functions as func
from BloodPressure.press import roi_press 

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        files = req.files.getlist("image")

        if not files:
            return func.HttpResponse("No image uploaded.", status_code=400)

        uploaded_file = files[0]

        # Read the uploaded image data as bytes
        image_data = uploaded_file.read()

        # Convert the image data to a numpy array
        image_np_array = np.frombuffer(image_data, dtype=np.uint8) 
        image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
        num=roi_press(image)
        headers = {
        "Content-type": "application/json",
        "Access-Control-Allow-Origin": "*"
    }
        return func.HttpResponse(json.dumps({
            'sys': num[0],
            'dia': num[1],
            'pulse':num[2]
            }),
            headers= headers,
            charset = 'utf-8',
            status_code=200
        )
    except ValueError:
         pass