from enum import Enum

CIFAR_CLASSES = [
    "airplanes",
    "cars",
    "birds",
    "cats",
    "deer",
    "dogs",
    "frogs",
    "horses",
    "ships",
    "trucks",
]
FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
MNIST_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class ResponseMessage(Enum):
    SUCCESSFULLY_PROCESSED_REQUEST = "Successfully processed request"
    ERROR_PROCESSING_REQUEST = "Error processing request"
