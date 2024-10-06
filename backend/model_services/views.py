import os

import numpy as np
import tensorflow as tf
from fastapi import APIRouter, UploadFile
from PIL import Image

from backend.model_services.constants import CIFAR_CLASSES

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


router = APIRouter()
cifar_model = tf.keras.models.load_model("static/cifar.h5", compile=False)
fashion_mnist_model = tf.keras.models.load_model(
    "static/fashion_mnist.h5", compile=False
)


@router.post("/cifar/predict")
async def predict_cifar(file: UploadFile):
    if not file:
        return {"error": "File is required!"}
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = Image.open(file.file).convert("RGB")
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = cifar_model.predict(image)
    predicted_class = CIFAR_CLASSES[np.argmax(prediction)]
    return {"image": predicted_class}


@router.post("/fashion_mnist/predict")
async def predict_fashion_mnist(file: UploadFile):
    if not file:
        return {"error": "File is required!"}
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = Image.open(file.file).convert("L")
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = fashion_mnist_model.predict(image)
    predicted_class = CIFAR_CLASSES[np.argmax(prediction)]
    return {"image": predicted_class}


@router.post("/mnist/digit/predict")
async def predict_mnist_digit(file: UploadFile):
    if file is None:
        # Handle the case where no file is provided
        return {"error": "File is required!"}
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = Image.open(file.file).convert("RGB")
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = cifar_model.predict(image)
    predicted_class = CIFAR_CLASSES[np.argmax(prediction)]
    return {"image": predicted_class}


@router.get("/get/all")
async def get_models():
    return {"models": ["cifar"]}
