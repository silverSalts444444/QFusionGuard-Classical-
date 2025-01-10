import numpy as np
import simulator
from skopt import gp_minimize
from skopt.space import Real

tgoal = 15

# Define the parameter search space
search_space = [
    Real(10, 20, name='temperature'),      # keV
    Real(5, 10, name='magnetic_strength'), # Tesla
    Real(0.01, 0.05, name='tritium_flow'), # grams/sec
    Real(40, 100, name='power_in'),        # MW
    Real(100, 500, name='coolant_flow')   # Liters/sec
]

# Define the objective function using the simulator
def objective(params):
    tempn,rad_loss,eff = params

    # Objective: Maximize efficiency, minimize radiation loss
    efficiency = eff
    penalty = rad_loss 
    td = abs(tempn-tgoal)
    return -efficiency + penalty + td # Maximize efficiency and penalize radiation losses and flutuation from ideal temp

# Run the optimization process
def optimize_fusion_reactor(state):
    result = gp_minimize(objective(state), search_space, n_calls=50, random_state=42)
    print("Optimal Parameters:", result.x)
    print("Maximum Efficiency:", -result.fun)
    return result.x
def run_loop():
    initial = (15,6,0.07,50,256)
    sim = simulator.Simulator(15,6,0.07,50,256)
    while True:
        outs = sim.simulate()
        iins = sim.get_inputs()
        np = optimize_fusion_reactor(outs)
        sim.update_inputs(np)

run_loop()