import os
import joblib
import pandas as pd
import numpy as np

def predict_shipment(input_data, model_path="src/congestion_model.pkl"):
    """
    Predicts congestion likelihood, confidence score, and key contributing factors.
    
    input_data: dict containing vessel_arrival_density, port_capacity, queue_length, 
               weather_condition ('Clear', 'Moderate', 'Severe'), historical_delay_avg
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run train_model.py first.")
        
    model = joblib.load(model_path)
    
    # Structure input data into DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Process categorical features matching training setup
    df_encoded = pd.get_dummies(df_input, columns=['weather_condition'])
    
    # Ensure all expected model columns exist
    expected_columns = [
        'vessel_arrival_density', 'port_capacity', 'queue_length', 
        'historical_delay_avg', 'weather_condition_Moderate', 'weather_condition_Severe'
    ]
    
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Align column ordering with training dataset
    X_single = df_encoded[expected_columns]
    
    # Get probability & prediction
    probabilities = model.predict_proba(X_single)[0]
    prediction = model.predict(X_single)[0]
    
    # High risk probability is index 1
    congestion_likelihood = probabilities[1]
    confidence_score = max(probabilities) * 100
    
    # Generate human-readable explanation based on feature contributions
    explanations = []
    if input_data['queue_length'] > 8:
        explanations.append(f"High port queue length ({input_data['queue_length']} ships waiting)")
    if input_data['vessel_arrival_density'] > input_data['port_capacity']:
        explanations.append(f"Arrival density ({input_data['vessel_arrival_density']}) exceeds port capacity ({input_data['port_capacity']})")
    if input_data['weather_condition'] in ['Moderate', 'Severe']:
        explanations.append(f"Adverse weather condition ({input_data['weather_condition']})")
    if input_data['historical_delay_avg'] > 3.0:
        explanations.append(f"High historical port delay average ({input_data['historical_delay_avg']:.1f} hrs)")
        
    if not explanations:
        explanations.append("Port operating under normal capacity and optimal weather conditions.")
        
    return {
        "is_congested": bool(prediction),
        "congestion_probability": round(float(congestion_likelihood), 4),
        "confidence_score_percent": round(float(confidence_score), 2),
        "key_contributing_factors": explanations
    }

if __name__ == "__main__":
    # Test sample prediction
    sample_shipment = {
        'vessel_arrival_density': 28,
        'port_capacity': 20,
        'queue_length': 10,
        'weather_condition': 'Severe',
        'historical_delay_avg': 4.5
    }
    
    print("--- Sample Inference Test ---")
    result = predict_shipment(sample_shipment)
    print(f"Congestion Flag: {result['is_congested']}")
    print(f"Probability: {result['congestion_probability']}")
    print(f"Confidence: {result['confidence_score_percent']}%")
    print("Key Drivers:")
    for driver in result['key_contributing_factors']:
        print(f" - {driver}")