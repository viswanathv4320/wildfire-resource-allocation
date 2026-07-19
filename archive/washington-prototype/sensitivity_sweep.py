import matplotlib.pyplot as plt
import numpy as np

def run_sensitivity_sweep():
    # Test Parameters
    wind_spd = 15 # mph
    wind_vec = 90 # North (90 degrees in our new system)
    base_r = 0.2
    
    # Range of c_factors to test
    c_values = [0.01, 0.03, 0.06, 0.1]
    angles = np.linspace(0, 360, 100)
    
    plt.figure(figsize=(10, 6))
    
    for c in c_values:
        probs = []
        for a in angles:
            # Use your existing logic
            diff = abs(wind_vec - a)
            mult = np.exp(c * wind_spd * np.cos(np.radians(diff)))
            probs.append(min(1.0, base_r * mult))
        
        plt.plot(angles, probs, label=f'c_factor: {c}')

    plt.title(f"Impact of c_factor on Spread Probability (Wind: {wind_spd}mph North)")
    plt.xlabel("Direction (Degrees: 0=E, 90=N, 180=W, 270=S)")
    plt.ylabel("Spread Probability")
    plt.xticks([0, 90, 180, 270], ['East', 'North', 'West', 'South'])
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()

run_sensitivity_sweep()