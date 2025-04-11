
# Visionify Backend 🧠🔍

This is the backend API for **Visionify**, a deep learning-powered image classification platform. Built using **FastAPI** and structured with **Clean Architecture** and **Dependency Injection**, this backend serves inference for multiple vision models with high modularity and scalability.

## 🚀 Features

- ⚡ Fast and asynchronous RESTful API using FastAPI
- 🧱 Clean architecture for clear separation of concerns
- 🧠 Model support:
  - MNIST Digit (10-class)
  - MNIST Fashion
  - CIFAR-10
- 🔌 Plug-and-play support for new models via DI
- 🧪 Fully testable modules
- 📦 Poetry for package and environment management
- ✅ Pre-commit hooks for consistent code quality

```

## 🔌 API Endpoints

| Endpoint                      | Method | Description                         |
|-------------------------------|--------|-------------------------------------|
| `/cifar/predict/`             | POST   | Predict using MNIST Digit model     |
| `/fashion/mnist/predict/`     | POST   | Predict using MNIST Fashion model   |
| `/mnist/digit/predict/`       | POST   | Predict using CIFAR-10 model        |

All endpoints expect a `multipart/form-data` request with an image file.

🧪 Test endpoints via:

```
http://localhost:8000/docs
```

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ShyamSundhar1411/Visionify-Backend.git
cd Visionify-Backend
```

### 2. Install dependencies using Poetry

```bash
poetry install
```

### 3. Run the server

```bash
poetry run uvicorn app.main:app --reload
```

Visit: [http://localhost:8000/docs](http://localhost:8000/docs)


## ✅ Code Quality

This project uses `pre-commit` for linting, formatting, and more.

### Setup pre-commit

```bash
pre-commit install
pre-commit run --all-files
```

## 📄 License

Licensed under the MIT License.

---

Made with ❤️ by [Shyam Sundhar](https://github.com/ShyamSundhar1411)
