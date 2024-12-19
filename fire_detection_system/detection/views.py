from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
from .models import UploadedImage  # Import the model

# Load the model globally when the server starts
model = load_model("C:/Users/Arren Bacarra/fire_detection_system/detection/model/fire_smoke_detection_model.h5")  # Update this with the actual model path

def upload_image(request):
    fire_detected = None
    media_url = None
    error = None

    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Get the uploaded image
            image = request.FILES['image']

            # Save the uploaded image to the file system
            fs = FileSystemStorage()
            file_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{image.name}"  # Add timestamp for unique filenames
            file_path = fs.save(file_name, image)
            file_url = fs.url(file_path)

            # Load and preprocess the image
            img = cv2.imread(fs.path(file_path))
            if img is None:
                raise ValueError("Invalid image file.")

            img = cv2.resize(img, (224, 224))  # Update size if model input differs
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            img = img / 255.0  # Normalize image (assuming model was trained with normalized images)
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Predict using the model
            prediction = model.predict(img)
            
            # Ensure the logic identifies fire correctly
            fire_detected = np.argmax(prediction, axis=-1)[0] == 1  # Class 1 is "fire"
            media_url = file_url

            # Save the image and result to the database
            uploaded_image = UploadedImage(
                image=file_name,  # Save the file name to the database
                fire_detected=fire_detected,  # Save the fire detection result
                uploaded_at=datetime.now()  # Save the upload timestamp
            )
            uploaded_image.save()  # Save to the database

        except Exception as e:
            error = str(e)  # Capture and display error messages in the template

    return render(request, 'upload.html', {
        'fire_detected': fire_detected,
        'media_url': media_url,
        'error': error,
    })
