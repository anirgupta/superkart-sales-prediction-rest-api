# Import necessary libraries
import numpy as np
import joblib                                       # For loading the serialized model pipeline
import pandas as pd                                 # For data manipulation
from flask import Flask, request, jsonify           # For creating the Flask REST API

# Initialize the Flask application
superkart_api = Flask("SuperKart Sales Predictor")

# Load the trained pipeline. This artifact contains the StandardScaler, the OneHotEncoder
# with its learned category vocabulary, and the tuned XGBoost model, so no separate
# preprocessing code is needed here.
model = joblib.load("superkart_model.joblib")

# The exact feature order the pipeline was trained on. Building the DataFrame from this
# list guarantees the columns arrive in the order the ColumnTransformer expects.
FEATURES = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Age_Years",
    "Product_Sugar_Content",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Product_Type_Category",
]


# Define a route for the home page (GET request)
@superkart_api.get('/')
def home():
    """
    Handles GET requests to the root URL ('/').
    Acts as a health check so that the container can be confirmed as running.
    """
    return "Welcome to the SuperKart Sales Prediction API!"


# Define an endpoint for single-product prediction (POST request)
@superkart_api.post('/v1/predict')
def predict_sales():
    """
    Handles POST requests to '/v1/predict'.
    Expects a JSON payload with the ten model features and returns the
    predicted sales figure as a JSON response.
    """
    try:
        # Get the JSON data from the request body
        product_data = request.get_json()

        # Extract the ten features the pipeline expects
        sample = {
            'Product_Weight': product_data['Product_Weight'],
            'Product_Allocated_Area': product_data['Product_Allocated_Area'],
            'Product_MRP': product_data['Product_MRP'],
            'Store_Age_Years': product_data['Store_Age_Years'],
            'Product_Sugar_Content': product_data['Product_Sugar_Content'],
            'Store_Size': product_data['Store_Size'],
            'Store_Location_City_Type': product_data['Store_Location_City_Type'],
            'Store_Type': product_data['Store_Type'],
            'Product_Id_char': product_data['Product_Id_char'],
            'Product_Type_Category': product_data['Product_Type_Category']
        }

        # Convert the extracted data into a single-row DataFrame
        input_data = pd.DataFrame([sample])[FEATURES]

        # Make the prediction. The pipeline scales and encodes before predicting.
        prediction = model.predict(input_data)[0]

        # Cast from numpy float32 to a native Python float. jsonify cannot serialize
        # numpy types and would otherwise raise a TypeError.
        prediction = round(float(prediction), 2)

        # Return the prediction as a JSON response
        return jsonify({'Predicted Sales (in dollars)': prediction})

    except KeyError as e:
        # A required field was absent from the payload
        return jsonify({'error': f'Missing required field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Define an endpoint for batch prediction (POST request)
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    """
    Handles POST requests to '/v1/predictbatch'.
    Expects a CSV file containing multiple product records and returns the
    predicted sales for each row, keyed by row index.
    """
    try:
        # Get the uploaded CSV file from the request
        file = request.files['file']

        # Read the CSV file into a DataFrame
        input_data = pd.read_csv(file)

        # Select the model features in the trained order. Any extra columns in the
        # uploaded file are ignored rather than causing a failure.
        input_data = input_data[FEATURES]

        # Make predictions for every row in the file
        predictions = model.predict(input_data).tolist()

        # Round each prediction and cast to a native Python float
        predictions = [round(float(p), 2) for p in predictions]

        # Build a dictionary keyed by row index, since the engineered feature set
        # does not carry a product identifier through to the model
        output_dict = dict(zip(input_data.index.astype(str), predictions))

        # Return the predictions dictionary as a JSON response
        return output_dict

    except KeyError as e:
        return jsonify({'error': f'Missing required column: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Run the Flask application in debug mode if this script is executed directly.
# In the container, gunicorn runs the app instead - see the Dockerfile.
if __name__ == '__main__':
    superkart_api.run(debug=True)
