import os
import numpy as np
import pandas as pd

def generate_port_data(num_records=300, output_path="data/port_data.csv"):
    np.random.seed(42)
    
    # Generate synthetic features
    vessel_arrival_density = np.random.randint(5, 35, size=num_records) # incoming ships per day
    port_capacity = np.random.choice([15, 20, 25, 30], size=num_records) # max handling capacity
    queue_length = np.random.randint(0, 15, size=num_records) # currently waiting
    weather_condition = np.random.choice(['Clear', 'Moderate', 'Severe'], size=num_records, p=[0.6, 0.3, 0.1])
    historical_delay_avg = np.random.uniform(0.5, 6.0, size=num_records) # hours
    
    data = pd.DataFrame({
        'vessel_arrival_density': vessel_arrival_density,
        'port_capacity': port_capacity,
        'queue_length': queue_length,
        'weather_condition': weather_condition,
        'historical_delay_avg': historical_delay_avg
    })
    
    # Map weather to risk multiplier
    weather_risk = data['weather_condition'].map({'Clear': 0, 'Moderate': 1.5, 'Severe': 3.5})
    
    # Define congestion ground truth logic: High queue, arrival density exceeding capacity, or severe weather
    congestion_score = (
        (data['queue_length'] > 8).astype(int) * 2 +
        (data['vessel_arrival_density'] > data['port_capacity']).astype(int) * 3 +
        weather_risk +
        (data['historical_delay_avg'] > 3.0).astype(int) * 1.5 +
        np.random.normal(0, 0.5, size=num_records) # noise
    )
    
    # Binary classification: 1 if congested/delayed > 4 hours, else 0
    data['is_congested'] = (congestion_score > 4.5).astype(int)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Dataset successfully generated and saved to {output_path} ({num_records} records).")

if __name__ == "__main__":
    generate_port_data()