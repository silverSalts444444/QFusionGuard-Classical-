import tensorflow as tf
from tensorflow.keras import *
import numpy as np
import simulator
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skopt import gp_minimize
from skopt.space import Real

# Define the model
inputd = 10
outputd = 4

mod = models.Sequential([
    layers.Dense(64, activation="relu", input_shape=(inputd,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(outputd)
])

mod.compile(optimizer="adam", loss="mse", metrics=["mae"])
def normalize_features(data):
    scaler = MinMaxScaler()  
    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler

def generate_data(num_samples, scaler=None):
    xt = []
    yt = []
    for i in range(num_samples):
        temp = np.random.uniform(10, 20)  # keV
        magstr = np.random.uniform(5, 10)  # Tesla
        tritflow = np.random.uniform(0.01, 0.05)  # grams/sec
        powerin = np.random.uniform(40, 100)  # MW
        powerout = np.random.uniform(30, 90)  # Estimated MW output
        h4flow = np.random.uniform(0.001, 0.01)  # Helium-4 flow rate
        neutronflux = np.random.uniform(1e12, 1e14)  # Neutrons/cm^2/s
        coolantflow = np.random.uniform(100, 500)  # Liters/sec
        fuelin = np.random.uniform(0.1, 0.5)  # grams/sec
        rad = np.random.uniform(1, 5)  # MW of radiation loss

        xt.append([temp, magstr, tritflow, powerin, powerout, h4flow, neutronflux, coolantflow, fuelin, rad])
        yt.append(simulator.Simulator(xt[i]).sim())

    xt = np.array(xt, dtype=np.float32)
    yt = np.array(yt, dtype=np.float32)

    if scaler is None:
        xt, scaler = normalize_features(xt)
    else:
        xt = scaler.transform(xt)  # Apply existing scaler to new data

    return xt, yt, scaler


# Initial data generation
xt, yt, scaler = generate_data(1000)

# Training the model with the initial dataset
history = mod.fit(xt, yt, epochs=50, batch_size=32, validation_split=0.2)

# Incremental training loop
def incremental_training(new_samples=200, epochs=10, scaler=None):
    for i in range(epochs):
        print(f"Training iteration {i+1}")
        
        # Generate new training data
        xt_new, yt_new, scaler = generate_data(new_samples, scaler=scaler)
        
        # Retrain the model with new data
        history = mod.fit(xt_new, yt_new, epochs=1, batch_size=32, validation_split=0.2)
        
        # Optionally, you can evaluate after each iteration
        loss, mae = mod.evaluate(xt_new, yt_new)
        print(f"Iteration {i+1} - Loss: {loss}, MAE: {mae}")
print(np.isnan(xt).any(), np.isnan(yt).any())
print(np.isinf(xt).any(), np.isinf(yt).any())
# Run incremental training
incremental_training(new_samples=200, epochs=10)

search_space = [
    Real(10, 20, name='temperature'),      # keV
    Real(5, 10, name='magnetic_strength'), # Tesla
    Real(0.01, 0.05, name='tritium_flow'), # grams/sec
    Real(40, 100, name='power_in'),        # MW
    Real(100, 500, name='coolant_flow'),   # Liters/sec
]

def objective(params):
    temp, magstr, tritflow, powerin, coolantflow = params
    inputs = np.array([[temp, magstr, tritflow, powerin, 0, 0, 0, coolantflow, 0, 0]])  # Fill in 0s for unused features
    normalized_inputs = scaler.transform(inputs)

    predictions = mod.predict(normalized_inputs)
    
    powerout = predictions[0][0]  
    rad_loss = predictions[0][3]  

    efficiency = powerout / powerin
    penalty = rad_loss 
    return -efficiency + penalty  

# Run optimization
def optimize_fusion_reactor():
    result = gp_minimize(objective, search_space, n_calls=50, random_state=42)
    print("Optimal Parameters:", result.x)
    print("Maximum Efficiency:", -result.fun)
    return result.x

# Call optimization after training is complete
if __name__ == "__main__":
    incremental_training(new_samples=200, epochs=10)
    optimal_params = optimize_fusion_reactor()