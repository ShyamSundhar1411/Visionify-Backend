import os

import numpy as np
from fastapi import APIRouter, Depends, UploadFile, status
from PIL import Image

from backend.model_services.constants import (
    CIFAR_CLASSES,
    FASHION_MNIST_CLASSES,
    MNIST_CLASSES,
    ResponseMessage,
)

from .dependencies import load_models
from .entities import ResponseBuilder

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


router = APIRouter()


@router.post("/cifar/predict/")
async def predict_cifar(file: UploadFile, models: tuple = Depends(load_models)):
    cifar_model, _, _ = models
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
    confidence_scores = {
        CIFAR_CLASSES[i]: float(prediction[0][i] * 100)
        for i in range(len(CIFAR_CLASSES))
    }
    data = {
        "predicted_class": predicted_class,
        "confidence_scores": confidence_scores,
    }
    return ResponseBuilder.build_response(
        data, ResponseMessage.SUCCESSFULLY_PROCESSED_REQUEST, status.HTTP_200_OK
    )


@router.post("/fashion/mnist/predict/")
async def predict_fashion_mnist(file: UploadFile, models: tuple = Depends(load_models)):
    _, fashion_mnist_model, _ = models
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
    predicted_class = FASHION_MNIST_CLASSES[np.argmax(prediction)]
    confidence_scores = {
        FASHION_MNIST_CLASSES[i]: float(prediction[0][i] * 100)
        for i in range(len(FASHION_MNIST_CLASSES))
    }

    data = {
        "predicted_class": predicted_class,
        "confidence_scores": confidence_scores,
    }
    return ResponseBuilder.build_response(
        data, ResponseMessage.SUCCESSFULLY_PROCESSED_REQUEST, status.HTTP_200_OK
    )


@router.post("/mnist/digit/predict/")
async def predict_mnist_digit(file: UploadFile, models: tuple = Depends(load_models)):
    _, _, digit_mnist_model = models
    if file is None:
        return {"error": "File is required!"}
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = Image.open(file.file).convert("L")
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = digit_mnist_model.predict(image)
    predicted_class = MNIST_CLASSES[np.argmax(prediction)]
    confidence_scores = {
        MNIST_CLASSES[i]: float(prediction[0][i] * 100)
        for i in range(len(MNIST_CLASSES))
    }

    data = {
        "predicted_class": predicted_class,
        "confidence_scores": confidence_scores,
    }
    return ResponseBuilder.build_response(
        data, ResponseMessage.SUCCESSFULLY_PROCESSED_REQUEST, status.HTTP_200_OK
    )


@router.get("/get/all/")
async def get_models():
    data = {"models": ["cifar", "fashion_mnist", "mnist_digit"]}
    return ResponseBuilder.build_response(
        data, ResponseMessage.SUCCESSFULLY_PROCESSED_REQUEST, status.HTTP_200_OK
    )
