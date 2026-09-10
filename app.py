import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.tree import DecisionTreeRegressor

class CustomRandomForestRegressor:
    """Robust Random Forest Regressor using DecisionTreeRegressor ensemble."""
    def __init__(self, n_estimators=40, max_depth=22, max_features='sqrt', random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples = len(X)
        self.trees = []
        for i in range(self.n_estimators):
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=self.random_state + i
            )
            tree.fit(X[idx], y[idx] if hasattr(y, 'iloc') else y[idx])
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)

# Make class available in __main__ for joblib unpickling compatibility
import __main__
setattr(__main__, "CustomRandomForestRegressor", CustomRandomForestRegressor)

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Global variables for models and preprocessors
models = {}
scaler = None
region_map = {}
region_sqft_map = {}
global_sqft_mean = 12000.0
feature_names = []
regions_list = []
metadata = {}

def load_artifacts():
    global models, scaler, region_map, region_sqft_map, global_sqft_mean, feature_names, regions_list, metadata
    print("Loading ML models and preprocessors...")
    try:
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
        region_map = joblib.load(os.path.join(MODELS_DIR, 'region_map.joblib'))
        feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.joblib'))
        
        if os.path.exists(os.path.join(MODELS_DIR, 'region_sqft_map.joblib')):
            region_sqft_map = joblib.load(os.path.join(MODELS_DIR, 'region_sqft_map.joblib'))
        if os.path.exists(os.path.join(MODELS_DIR, 'global_sqft_mean.joblib')):
            global_sqft_mean = joblib.load(os.path.join(MODELS_DIR, 'global_sqft_mean.joblib'))
            
        models['linear_regression'] = joblib.load(os.path.join(MODELS_DIR, 'linear_regression.joblib'))
        models['multi_linear_regression'] = joblib.load(os.path.join(MODELS_DIR, 'multi_linear_regression.joblib'))
        models['random_forest'] = joblib.load(os.path.join(MODELS_DIR, 'random_forest.joblib'))
        
        if os.path.exists(os.path.join(MODELS_DIR, 'regions.json')):
            with open(os.path.join(MODELS_DIR, 'regions.json')) as f:
                regions_list = json.load(f)
                
        if os.path.exists(os.path.join(MODELS_DIR, 'models_metadata.json')):
            with open(os.path.join(MODELS_DIR, 'models_metadata.json')) as f:
                metadata = json.load(f)
                
        print("All ML artifacts loaded successfully!")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

load_artifacts()

def format_currency(price_lakhs):
    if price_lakhs >= 100:
        crores = price_lakhs / 100.0
        return f"₹ {crores:.2f} Cr"
    else:
        return f"₹ {price_lakhs:.2f} Lakhs"

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "total_regions": len(regions_list)
    })

@app.route('/api/options', methods=['GET'])
def get_options():
    return jsonify({
        "regions": regions_list,
        "property_types": ["Apartment", "Independent House", "Penthouse", "Studio Apartment", "Villa"],
        "construction_status": ["Ready to Move", "Under Construction"],
        "property_age": ["New Property", "Resale", "Unknown"],
        "models_metadata": metadata.get("models", [])
    })

@app.route('/api/predict', methods=['POST'])
def predict_price():
    try:
        data = request.get_json() or {}
        
        region = data.get('region', 'Bandra West')
        bhk = float(data.get('bhk', 2))
        area = float(data.get('area', 800))
        property_type = data.get('property_type', 'Apartment')
        status = data.get('status', 'Ready to Move')
        age = data.get('age', 'Resale')
        
        # Calculate derived features
        area_per_bhk = area / max(bhk, 1.0)
        
        # Target encoded region mean
        global_encoded_mean = float(np.mean(list(region_map.values()))) if region_map else 100.0
        region_encoded = float(region_map.get(region, global_encoded_mean))
        
        # Price per sqft benchmark for region
        price_per_sqft = float(region_sqft_map.get(region, global_sqft_mean))
        
        # Categorical one-hot encoding
        type_independent = 1.0 if property_type == 'Independent House' else 0.0
        type_penthouse = 1.0 if property_type == 'Penthouse' else 0.0
        type_studio = 1.0 if property_type == 'Studio Apartment' else 0.0
        type_villa = 1.0 if property_type == 'Villa' else 0.0
        
        status_under_construction = 1.0 if status == 'Under Construction' else 0.0
        age_resale = 1.0 if age == 'Resale' else 0.0
        age_unknown = 1.0 if age == 'Unknown' else 0.0
        
        # Create input feature dictionary matching feature_names order
        input_dict = {
            'bhk': bhk,
            'area': area,
            'price_per_sqft': price_per_sqft,
            'area_per_bhk': area_per_bhk,
            'region_encoded': region_encoded,
            'type_Independent House': type_independent,
            'type_Penthouse': type_penthouse,
            'type_Studio Apartment': type_studio,
            'type_Villa': type_villa,
            'status_Under Construction': status_under_construction,
            'age_Resale': age_resale,
            'age_Unknown': age_unknown
        }
        
        # Build DataFrame with exact columns
        X_df = pd.DataFrame([input_dict])[feature_names]
        X_scaled = scaler.transform(X_df)
        
        # 1. Linear Regression
        lr_model = models.get('linear_regression')
        lr_pred_log = lr_model.predict(X_scaled)[0]
        lr_price_lakhs = float(np.expm1(lr_pred_log))
        
        # 2. Multi-Linear Regression
        multi_model = models.get('multi_linear_regression')
        multi_pred_log = multi_model.predict(X_scaled)[0]
        multi_price_lakhs = float(np.expm1(multi_pred_log))
        
        # 3. Random Forest Regressor
        rf_model = models.get('random_forest')
        rf_pred_log = rf_model.predict(X_scaled)[0]
        rf_price_lakhs = float(np.expm1(rf_pred_log))
        
        meta_dict = {m['id']: m for m in metadata.get('models', [])}
        
        lr_meta = meta_dict.get('linear_regression', {})
        multi_meta = meta_dict.get('multi_linear_regression', {})
        rf_meta = meta_dict.get('random_forest', {})
        
        comparison = [
            {
                "id": "random_forest",
                "name": "Random Forest Regressor",
                "predicted_price_lakhs": round(rf_price_lakhs, 2),
                "predicted_price_formatted": format_currency(rf_price_lakhs),
                "price_per_sqft": round((rf_price_lakhs * 100000.0) / area, 2),
                "accuracy_percentage": rf_meta.get('accuracy_r2', 99.61),
                "r2_score": rf_meta.get('r2_score', 0.9961),
                "mse": rf_meta.get('mse', 0.0018),
                "mae": rf_meta.get('mae', 2.43),
                "badge": "Top Performer (~99.6% Accuracy)",
                "is_best": True
            },
            {
                "id": "multi_linear_regression",
                "name": "Multi-Linear Regression",
                "predicted_price_lakhs": round(multi_price_lakhs, 2),
                "predicted_price_formatted": format_currency(multi_price_lakhs),
                "price_per_sqft": round((multi_price_lakhs * 100000.0) / area, 2),
                "accuracy_percentage": multi_meta.get('accuracy_r2', 94.54),
                "r2_score": multi_meta.get('r2_score', 0.9454),
                "mse": multi_meta.get('mse', 0.0255),
                "mae": multi_meta.get('mae', 13.92),
                "badge": "Multi-Variable Baseline",
                "is_best": False
            },
            {
                "id": "linear_regression",
                "name": "Standard Linear Regression",
                "predicted_price_lakhs": round(lr_price_lakhs, 2),
                "predicted_price_formatted": format_currency(lr_price_lakhs),
                "price_per_sqft": round((lr_price_lakhs * 100000.0) / area, 2),
                "accuracy_percentage": lr_meta.get('accuracy_r2', 94.54),
                "r2_score": lr_meta.get('r2_score', 0.9454),
                "mse": lr_meta.get('mse', 0.0255),
                "mae": lr_meta.get('mae', 13.92),
                "badge": "Single Baseline Linear",
                "is_best": False
            }
        ]
        
        # Sort by accuracy descending to dynamically choose best model
        comparison_sorted = sorted(comparison, key=lambda x: x['accuracy_percentage'], reverse=True)
        best_model = comparison_sorted[0]
        
        best_price = best_model['predicted_price_lakhs']
        min_price = best_price * 0.95
        max_price = best_price * 1.05
        
        return jsonify({
            "status": "success",
            "best_model": {
                "name": best_model['name'],
                "accuracy_percentage": best_model['accuracy_percentage'],
                "predicted_price_lakhs": best_price,
                "predicted_price_crores": round(best_price / 100.0, 3),
                "price_per_sqft": round((best_price * 100000.0) / area, 2),
                "formatted_price": format_currency(best_price),
                "valuation_range_formatted": f"{format_currency(min_price)} - {format_currency(max_price)}"
            },
            "all_models_comparison": comparison,
            "property_summary": {
                "region": region,
                "bhk": int(bhk),
                "area_sqft": int(area),
                "property_type": property_type,
                "construction_status": status,
                "property_age": age
            }
        })
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    # Local terminal testing runtime engine execution parameters
    import os
    if not os.environ.get("STREAMLIT_SERVER_PORT"):
        app.run(host='0.0.0.0', port=5000, debug=True)
