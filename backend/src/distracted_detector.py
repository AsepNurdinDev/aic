import numpy as np
import tensorflow as tf


class DistractedDetector:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

        self.classes = {
            0: {
                "class": "c0",
                "label": "Normal Driving",
                "status": "NORMAL"
            },
            1: {
                "class": "c5",
                "label": "Operating the Radio",
                "status": "DISTRACTED"
            },
            2: {
                "class": "c6",
                "label": "Drinking",
                "status": "DISTRACTED"
            }
        }

    def predict(self, frame_bgr):
        # OpenCV menggunakan BGR.
        # Model membutuhkan RGB.
        frame_rgb = frame_bgr[:, :, ::-1]

        # Resize ke ukuran input model
        image = tf.image.resize(
            frame_rgb,
            (224, 224)
        )

        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, axis=0)

        # Prediksi
        prediction = self.model.predict(
            image,
            verbose=0
        )[0]

        predicted_index = int(np.argmax(prediction))
        confidence = float(prediction[predicted_index])

        result = self.classes[predicted_index]

        return {
            "class": result["class"],
            "label": result["label"],
            "status": result["status"],
            "confidence": round(confidence, 4)
        }