import numpy as np
from skopt import gp_minimize
from skopt.space import Real
import simulator  # Assuming this is your custom simulator

tgoal = 15  # Target temperature

# Define the parameter search space
search_space = [
    Real(10, 20, name='temperature'),      # keV
    Real(5, 10, name='magnetic_strength'), # Tesla
    Real(0.01, 0.05, name='tritium_flow'), # grams/sec
    Real(40, 100, name='power_in'),        # MW
    Real(100, 500, name='coolant_flow')    # Liters/sec
]

# Define the optimization function
def optimize_fusion_reactor(current_outputs, sim):
    """
    Optimize the fusion reactor's inputs to bring the system back on track.
    
    Parameters:
    - current_outputs: tuple (tempn, rad_loss, eff), the current state of the reactor.
    - sim: The simulator instance.
    
    Returns:
    - optimal_params: List of optimized parameters.
    """
    def wrapped_objective(params):
        """
        Wraps the simulator's inputs for optimization purposes.
        Simulates the system with given parameters and evaluates the objective.
        """
        sim.update_inputs(*params)  # Update the simulator with current trial parameters
        tempn, rad_loss, eff = sim.simulate()  # Run simulation to get new outputs
        
        # Calculate objective as defined
        efficiency = eff
        penalty = rad_loss
        temp_deviation = abs(tempn - tgoal)
        return 10*-efficiency + penalty + temp_deviation

    # Run optimization
    result = gp_minimize(
        wrapped_objective,
        search_space,
        n_calls=10,  # Reduce calls to make it faster for continuous updates
        random_state=42
    )
    return result.x  # Return the optimized parameters

# Run the optimization loop
def run_loop():
    # Initial reactor parameters
    initial_params = (15, 6, 1, 50, 2)
    sim = simulator.Simulator(*initial_params)

    while True:
        # Simulate the current state of the reactor
        current_outputs = sim.simulate()  # Outputs should match the expected structure
        print(f"Current Simulation Outputs: {current_outputs}")

        # Optimize the reactor parameters based on the current state
        optimized_params = optimize_fusion_reactor(current_outputs, sim)
        print("Optimized Parameters:", optimized_params)

        # Update the simulator with the optimized parameters
        sim.update_inputs(*optimized_params)

        # Optionally, add a break condition or delay for real-world scenarios
        # e.g., time.sleep(1) for periodic updates

# Start the simulation loop
run_loop()
