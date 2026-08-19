# 🏠 Property Price Prediction

A machine learning-powered web application that predicts the estimated market value of a property based on its characteristics such as number of bedrooms, bathrooms, living area, lot area, location, condition, year built, and other property-related features.

The project combines a trained machine learning model with a Flask backend and a responsive web interface. Users can enter property information through the frontend, submit the details to the backend, and receive the predicted property value in real time.

---

## 📌 Project Overview

Property price prediction is a practical machine learning problem that can help estimate the market value of residential properties using historical property data.

In this project, a machine learning model is trained on historical housing/property information. The trained model is saved as a `.pkl` file and integrated into a Flask web application.

The application provides a simple interface where users can:

- Enter property characteristics
- Select location information
- Submit the property details
- Send the information to the Flask backend
- Generate a machine learning prediction
- View the estimated property value prominently on the webpage

The application is designed to be deployment-ready and can be hosted using cloud platforms such as Render.

---

## 🎯 Project Objectives

The main objectives of this project are:

1. To develop a machine learning model for predicting property prices.
2. To use historical property data for training the prediction model.
3. To save the trained model for later inference.
4. To integrate the trained model with a Flask backend.
5. To create a user-friendly web interface for entering property details.
6. To provide real-time predictions through a web API.
7. To display the predicted property value in a clear and professional interface.
8. To make the application suitable for cloud deployment.
9. To demonstrate the practical integration of machine learning with web development.

---

## ✨ Key Features

### 🤖 Machine Learning

- Trained machine learning model for property price prediction
- Saved model using Python serialization
- Model loaded during application startup
- Prediction performed using user-provided property features

### 🌐 Web Application

- Flask-based backend
- REST-style `/predict` endpoint
- HTML, CSS, and JavaScript frontend
- Asynchronous prediction requests using JavaScript
- Real-time prediction display
- Professional responsive user interface

### 🏡 Property Information

The application supports property-related inputs such as:

- Bedrooms
- Bathrooms
- Living area
- Lot area
- Number of floors
- Waterfront availability
- View rating
- Property condition
- Above-ground area
- Basement area
- Year built
- Year renovated
- City
- Country
- Month
- Day
- Year

---

# 🧠 Machine Learning Workflow

The overall machine learning workflow can be represented as:

```text
Historical Property Dataset
          │
          ▼
   Data Preprocessing
          │
          ▼
   Feature Preparation
          │
          ▼
    Model Training
          │
          ▼
   Model Evaluation
          │
          ▼
   Save Trained Model
        (.pkl)
          │
          ▼
    Flask Integration
          │
          ▼
    User Input
          │
          ▼
      Prediction
          │
          ▼
 Estimated Property Value
