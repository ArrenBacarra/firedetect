from django.db import models
from django.utils.timezone import now

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploaded_images/')
    fire_detected = models.BooleanField(default=False)  # Field to store fire detection result
    uploaded_at = models.DateTimeField(default=now)  # Field to store the timestamp of the upload

    def __str__(self):
        return f"Image uploaded on {self.uploaded_at}, Fire Detected: {self.fire_detected}"
