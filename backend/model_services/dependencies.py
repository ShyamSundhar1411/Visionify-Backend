import asyncio

import tensorflow as tf
from fastapi import HTTPException


async def load_models():
    try:
        cifar_model = await asyncio.to_thread(
            tf.keras.models.load_model, "static/cifar.h5", compile=False
        )
        fashion_mnist_model = await asyncio.to_thread(
            tf.keras.models.load_model, "static/fashion_mnist.h5", compile=False
        )
        digit_mnist_model = await asyncio.to_thread(
            tf.keras.models.load_model, "static/mnist.h5", compile=False
        )
        return cifar_model, fashion_mnist_model, digit_mnist_model
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
