# I upgraded to python 3.10 to be able to use tensorflow for our CNN

# To create a python3.10 env use: conda create -n "python310env" python=3.10
# To download all required packages use: pip install fastapi uvicorn motor pymongo joblib numpy scikit-learn tensorflow-macos Pillow pydantic pydantic-core
# Then do the following:
# 1st, To enter terminal use: conda activate python310env
# 2nd, To start the mongodb server: brew services start mongodb-community@6.0
# 3rd, To start the server: uvicorn server:app --host 0.0.0.0 --port 8000
# 4th, To stop the mongodb server: brew services stop mongodb-community@6.0

# To connect to the server go to: http://localhost:8000/docs


import os
import logging
import base64
from io import BytesIO
from typing import Optional, List

import numpy as np
from PIL import Image
from bson import ObjectId
from fastapi import FastAPI, Body, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from pydantic import GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

import motor.motor_asyncio
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
import tensorflow as tf
from tensorflow.keras import layers, models

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom lifespan function to initialize resources
async def custom_lifespan(app: FastAPI):
    # Initialize MongoDB client
    app.mongo_client = motor.motor_asyncio.AsyncIOMotorClient()
    db = app.mongo_client.ml_database
    app.collection = db.get_collection("labeled_instances")

    # Initialize models storage
    app.knn_models = {}
    app.cnn_models = {}

    # Initialize current model type and dsid
    app.current_model_type = "KNN"
    app.current_dsid = 0

    yield

    # Close MongoDB client
    app.mongo_client.close()

# Initialize FastAPI app
app = FastAPI(
    title="Machine Learning as a Service",
    description="An application using FastAPI to provide ML services.",
    lifespan=custom_lifespan,
)

# Configure CORS (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the PyObjectId class compatible with Pydantic v2
class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.general_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str) and ObjectId.is_valid(v):
            return ObjectId(v)
        raise ValueError('Invalid ObjectId')

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        json_schema = handler(core_schema)
        json_schema.update(type='string')
        return json_schema

# Define Pydantic models
class LabeledDataPoint(BaseModel):
    id: Optional[PyObjectId] = Field(alias='_id')
    image_data: str  # Base64 encoded image
    label: int       # Digit label (0-9)
    dsid: int        # Dataset ID

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "image_data": "base64_encoded_string",
                "label": 5,
                "dsid": 1
            }
        }
    )

class FeatureDataPoint(BaseModel):
    image_data: str  # Base64 encoded image
    dsid: int        # Dataset ID

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image_data": "base64_encoded_string",
                "dsid": 1
            }
        }
    )

class ModelSelection(BaseModel):
    model_type: str  # "KNN" or "CNN"
    dsid: int        # Dataset ID

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_type": "CNN",
                "dsid": 1
            }
        }
    )

# Endpoint to upload labeled data
@app.post("/upload_data/", status_code=status.HTTP_201_CREATED)
async def upload_data(data: LabeledDataPoint):
    """
    Endpoint to upload a labeled data point.
    """
    # Insert data into MongoDB
    doc = data.dict(by_alias=True)
    result = await app.collection.insert_one(doc)
    logger.info(f"Inserted data with id {result.inserted_id}")
    return {"status": "Data received", "id": str(result.inserted_id)}

# Endpoint to set the model type
@app.post("/set_model/")
async def set_model(selection: ModelSelection):
    """
    Endpoint to set the current model type (KNN or CNN) for a specific DSID.
    """
    app.current_model_type = selection.model_type.upper()
    app.current_dsid = selection.dsid
    logger.info(f"Model set to {app.current_model_type} for DSID {app.current_dsid}")
    return {"status": f"Model set to {app.current_model_type} for DSID {app.current_dsid}"}

# Endpoint to train the model
@app.get("/train_model/{dsid}")
async def train_model(dsid: int):
    """
    Endpoint to train the model (KNN or CNN) for the specified DSID.
    """
    # Retrieve data from MongoDB
    datapoints = await app.collection.find({"dsid": dsid}).to_list(length=None)
    if len(datapoints) < 2:
        raise HTTPException(status_code=400, detail="Not enough data to train the model.")

    # Prepare data
    images = []
    labels = []
    for dp in datapoints:
        img_data = base64.b64decode(dp["image_data"])
        image = Image.open(BytesIO(img_data)).convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image) / 255.0  # Normalize pixel values
        images.append(image_array)
        labels.append(dp["label"])
    images = np.array(images)
    labels = np.array(labels)

    if app.current_model_type == "KNN":
        # Prepare data for KNN
        images_flat = images.reshape(len(images), -1)
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(images_flat, labels)
        yhat = model.predict(images_flat)
        acc = np.sum(yhat == labels) / len(labels)

        # Save model
        model_file_path = f'models/knn_model_dsid{dsid}.joblib'
        os.makedirs('models', exist_ok=True)
        dump(model, model_file_path)
        app.knn_models[dsid] = model

        logger.info(f"KNN model trained for DSID {dsid} with accuracy {acc}")
        return {"summary": f"KNN model trained with accuracy {acc}"}

    elif app.current_model_type == "CNN":
        # Prepare data for CNN
        images = images.reshape(-1, 28, 28, 1)
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(images, labels, epochs=5, verbose=1)
        loss, acc = model.evaluate(images, labels, verbose=0)

        # Save model
        model_file_path = f'models/cnn_model_dsid{dsid}.h5'
        os.makedirs('models', exist_ok=True)
        model.save(model_file_path)
        app.cnn_models[dsid] = model

        logger.info(f"CNN model trained for DSID {dsid} with accuracy {acc}")
        return {"summary": f"CNN model trained with accuracy {acc}"}

    else:
        raise HTTPException(status_code=400, detail="Invalid model type selected.")

# Endpoint to make predictions
@app.post("/predict/")
async def predict(data: FeatureDataPoint):
    """
    Endpoint to predict the label of an image using the selected model.
    """
    dsid = data.dsid
    img_data = base64.b64decode(data.image_data)
    image = Image.open(BytesIO(img_data)).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0

    if app.current_model_type == "KNN":
        # Load model if not already loaded
        if dsid not in app.knn_models:
            try:
                model_file_path = f'models/knn_model_dsid{dsid}.joblib'
                model = load(model_file_path)
                app.knn_models[dsid] = model
                logger.info(f"KNN model loaded for DSID {dsid}")
            except:
                raise HTTPException(status_code=404, detail="KNN model not found for this DSID.")
        else:
            model = app.knn_models[dsid]

        # Prepare data
        image_flat = image_array.reshape(1, -1)
        prediction = model.predict(image_flat)
        logger.info(f"KNN prediction for DSID {dsid}: {prediction[0]}")
        return {"prediction": int(prediction[0])}

    elif app.current_model_type == "CNN":
        # Load model if not already loaded
        if dsid not in app.cnn_models:
            try:
                model_file_path = f'models/cnn_model_dsid{dsid}.h5'
                model = tf.keras.models.load_model(model_file_path)
                app.cnn_models[dsid] = model
                logger.info(f"CNN model loaded for DSID {dsid}")
            except:
                raise HTTPException(status_code=404, detail="CNN model not found for this DSID.")
        else:
            model = app.cnn_models[dsid]

        # Prepare data
        image_input = image_array.reshape(1, 28, 28, 1)
        predictions = model.predict(image_input)
        prediction = np.argmax(predictions, axis=1)
        logger.info(f"CNN prediction for DSID {dsid}: {prediction[0]}")
        return {"prediction": int(prediction[0])}

    else:
        raise HTTPException(status_code=400, detail="Invalid model type selected.")

# Endpoint to evaluate the model
@app.get("/evaluate/{dsid}")
async def evaluate_model(dsid: int):
    """
    Endpoint to evaluate the current model's performance on the dataset.
    """
    # Retrieve data from MongoDB
    datapoints = await app.collection.find({"dsid": dsid}).to_list(length=None)
    if len(datapoints) < 2:
        raise HTTPException(status_code=400, detail="Not enough data to evaluate the model.")

    # Prepare data
    images = []
    labels = []
    for dp in datapoints:
        img_data = base64.b64decode(dp["image_data"])
        image = Image.open(BytesIO(img_data)).convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image) / 255.0
        images.append(image_array)
        labels.append(dp["label"])
    images = np.array(images)
    labels = np.array(labels)

    if app.current_model_type == "KNN":
        # Load model
        if dsid not in app.knn_models:
            try:
                model_file_path = f'models/knn_model_dsid{dsid}.joblib'
                model = load(model_file_path)
                app.knn_models[dsid] = model
                logger.info(f"KNN model loaded for evaluation DSID {dsid}")
            except:
                raise HTTPException(status_code=404, detail="KNN model not found for this DSID.")
        else:
            model = app.knn_models[dsid]

        # Prepare data
        images_flat = images.reshape(len(images), -1)
        yhat = model.predict(images_flat)
        acc = np.sum(yhat == labels) / len(labels)
        logger.info(f"KNN model accuracy for DSID {dsid}: {acc}")

        return {"model": "KNN", "accuracy": acc}

    elif app.current_model_type == "CNN":
        # Load model
        if dsid not in app.cnn_models:
            try:
                model_file_path = f'models/cnn_model_dsid{dsid}.h5'
                model = tf.keras.models.load_model(model_file_path)
                app.cnn_models[dsid] = model
                logger.info(f"CNN model loaded for evaluation DSID {dsid}")
            except:
                raise HTTPException(status_code=404, detail="CNN model not found for this DSID.")
        else:
            model = app.cnn_models[dsid]

        # Prepare data
        images = images.reshape(-1, 28, 28, 1)
        loss, acc = model.evaluate(images, labels, verbose=0)
        logger.info(f"CNN model accuracy for DSID {dsid}: {acc}")

        return {"model": "CNN", "accuracy": acc, "loss": loss}

    else:
        raise HTTPException(status_code=400, detail="Invalid model type selected.")