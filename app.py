from flask import Flask, render_template, request, jsonify, send_from_directory
import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime
import logging

# Initialize Flask app
app = Flask(__name__)   

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
feature_columns = [
    'Cereals and products',
    'Meat and fish', 
    'Egg',
    'Milk and products',
    'Oils and fats',
    'Fruits',
    'Vegetables',
    'Pulses and products',
    'Sugar and Confectionery',
    'Spices',
    'Non-alcoholic beverages',
    'Prepared meals, snacks, sweets etc.',
    'Food and beverages',
    'Pan, tobacco and intoxicants',
    'Clothing',
    'Footwear',
    'Clothing and footwear',
    'Housing',
    'Fuel and light',
    'Household goods and services',
    'Health',
    'Transport and communication',
    'Recreation and amusement',
    'Education',
    'Personal care and effects',
    'Miscellaneous',
    'Rural',
    'Urban',
    'Combined'
]

def load_model():
    """Load the trained model from pickle file"""
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model file not found")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model = DummyModel()

class DummyModel:
    """Dummy model for demonstration when actual model is not available"""
    
    def __init__(self):
        # Sample weights for demonstration
        self.weights = np.array([
            0.120, 0.089, 0.045, 0.098, 0.067, 0.076, 0.134, 0.052, 0.034, 0.028,
            0.045, 0.089, 0.456, 0.023, 0.067, 0.021, 0.088, 0.089, 0.078, 0.056,
            0.067, 0.089, 0.034, 0.045, 0.038, 0.298, 0.334, 0.367, 0.299
        ])
        self.intercept = 45.0
    
    def predict(self, X):
        """Predict using dummy weighted sum"""
        if X.shape[1] != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} features, got {X.shape[1]}")
        
        # Simple weighted sum prediction
        predictions = np.dot(X, self.weights) * 0.85 + self.intercept
        return predictions

def validate_input_data(data):
    """Validate input data for prediction"""
    errors = []
    
    # Check if all required fields are present
    for column in feature_columns:
        if column not in data:
            errors.append(f"Missing field: {column}")
    
    # Check if values are numeric and within reasonable range
    for column in feature_columns:
        if column in data:
            try:
                value = float(data[column])
                if value < 0 or value > 1000:  # Reasonable range for CPI values
                    errors.append(f"Value for {column} is out of range (0-1000): {value}")
            except ValueError:
                errors.append(f"Invalid numeric value for {column}: {data[column]}")
    
    return errors

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get data from request
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        logger.info(f"Received prediction request with {len(data)} parameters")
        
        # Validate input data
        validation_errors = validate_input_data(data)
        if validation_errors:
            return jsonify({
                'error': 'Validation failed',
                'details': validation_errors
            }), 400
        
        # Prepare input array
        input_values = []
        for column in feature_columns:
            input_values.append(float(data[column]))
        
        input_array = np.array(input_values).reshape(1, -1)
        
        # Make prediction
        if model is None:
            load_model()
        
        prediction = model.predict(input_array)[0]
        
        # Round to 2 decimal places
        prediction = round(float(prediction), 2)
        
        logger.info(f"Prediction made successfully: {prediction}")
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat(),
            'model_type': type(model).__name__
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500

@app.route('/api/features')
def get_features():
    """Get list of feature columns"""
    return jsonify({
        'features': feature_columns,
        'count': len(feature_columns)
    })

@app.route('/api/model-info')
def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        load_model()
    
    return jsonify({
        'model_type': type(model).__name__,
        'feature_count': len(feature_columns),
        'features': feature_columns,
        'is_dummy': isinstance(model, DummyModel)
    })

@app.route('/api/sample-data')
def get_sample_data():
    """Get sample data for testing"""
    sample_data = {
        'Cereals and products': 145.2,
        'Meat and fish': 178.9,
        'Egg': 167.4,
        'Milk and products': 156.8,
        'Oils and fats': 134.5,
        'Fruits': 189.3,
        'Vegetables': 201.7,
        'Pulses and products': 123.8,
        'Sugar and Confectionery': 145.6,
        'Spices': 167.2,
        'Non-alcoholic beverages': 134.9,
        'Prepared meals, snacks, sweets etc.': 156.7,
        'Food and beverages': 167.8,
        'Pan, tobacco and intoxicants': 189.4,
        'Clothing': 145.6,
        'Footwear': 156.2,
        'Clothing and footwear': 148.9,
        'Housing': 167.3,
        'Fuel and light': 178.5,
        'Household goods and services': 145.7,
        'Health': 189.2,
        'Transport and communication': 156.8,
        'Recreation and amusement': 167.4,
        'Education': 178.9,
        'Personal care and effects': 145.3,
        'Miscellaneous': 167.6,
        'Rural': 156.8,
        'Urban': 167.4,
        'Combined': 162.1
    }
    
    return jsonify(sample_data)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Not found',
        'message': 'The requested resource was not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'Something went wrong on our end'
    }), 500

# Initialize model on startup
# @app.before_first_request
# def initialize():
#     """Initialize the application"""
#     logger.info("Initializing CPI Prediction App")
load_model()

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Load model
    load_model()
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting CPI Prediction App on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)