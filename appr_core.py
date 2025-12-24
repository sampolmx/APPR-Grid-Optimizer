import numpy as np
import pandas as pd

# Constants (moved from notebook for consistency)
CAPACIDAD_BATERIA_MWh = 60.0
TASA_MAX_MW = 20.0
SOC_INICIAL = CAPACIDAD_BATERIA_MWh / 2

def enrich_data_with_forecast(df, forecast_error_std=5.0):
    """
    Adds a 'Forecast_Solar_MW' column to the dataframe.
    Simulates a forecast by adding noise to the actual generation.
    """
    df = df.copy()
    # Ensure Generacion_Solar_MW exists
    if 'Generacion_Solar_MW' not in df.columns:
        raise ValueError("DataFrame must contain 'Generacion_Solar_MW'")

    # Simulate forecast = Actual + Noise
    noise = np.random.normal(0, forecast_error_std, len(df))
    df['Forecast_Solar_MW'] = df['Generacion_Solar_MW'] + noise
    
    # Clip to be realistic (non-negative)
    df['Forecast_Solar_MW'] = df['Forecast_Solar_MW'].clip(lower=0)
    
    return df

class GridEnvironment:
    def __init__(self, df, capacity_limit, battery_capacity, battery_rate, training_mode=True):
        self.df = df
        
        # Ensure forecast column exists
        if 'Forecast_Solar_MW' not in df.columns:
             # Fallback if not present, though ideally should be enriched first
             print("Warning: 'Forecast_Solar_MW' not found. Using 'Generacion_Solar_MW' as perfect forecast.")
             self.df = df.copy()
             self.df['Forecast_Solar_MW'] = self.df['Generacion_Solar_MW']

        # Convert to numpy for performance
        # Columns: [Generacion, Demanda, Forecast]
        self.data_matrix = self.df[['Generacion_Solar_MW', 'Demanda_MW', 'Forecast_Solar_MW']].values
        
        self.limit = capacity_limit
        self.battery_mwh = battery_capacity
        self.battery_rate = battery_rate
        self.training_mode = training_mode
        
        # Actions: 0=Hold, 1=Charge, 2=Discharge
        self.action_space_size = 3
        
        # State: [Solar_MW, Demanda_MW, SOC_t, Forecast_Solar_T+1]
        self.state_space_size = 4 
        self.soc = SOC_INICIAL
        self.reset()

    def reset(self):
        self.current_step = 0
        self.soc = SOC_INICIAL 
        self.done = False
        return self._get_state()

    def _get_state(self):
        if self.current_step >= len(self.df):
            return np.zeros(self.state_space_size)
            
        # Get forecast for T+1
        # If we are at the last step, there is no T+1, assume 0 or repeat T
        forecast_next = 0.0
        if self.current_step + 1 < len(self.df):
            forecast_next = self.data_matrix[self.current_step + 1, 2] # Index 2 is Forecast
            
        state = [
            self.data_matrix[self.current_step, 0], # Generacion_Solar_MW
            self.data_matrix[self.current_step, 1], # Demanda_MW
            self.soc,
            forecast_next                           # Forecast for next hour
        ]
        return np.array(state)

    def step(self, action):
        # Action: 0=Hold, 1=Charge, 2=Discharge
        
        power_action_MW = 0
        if action == 1: # Charge
            power_action_MW = self.battery_rate
        elif action == 2: # Discharge
            power_action_MW = -self.battery_rate
        
        G_t = self.data_matrix[self.current_step, 0]
        
        curtailment_t = 0
        injection_t = 0
        
        # --- Dispatch Logic ---
        
        # A. Excess Generation (G_t > Limit)
        if G_t > self.limit:
            exceso_a_manejar = G_t - self.limit
            
            if action == 1: # Wants to Charge
                charge_amount = min(exceso_a_manejar, power_action_MW, self.battery_mwh - self.soc)
                self.soc += charge_amount
                curtailment_t = exceso_a_manejar - charge_amount
            else:
                curtailment_t = exceso_a_manejar
            
            injection_t = self.limit
            
        # B. No Excess (G_t <= Limit)
        else:
            curtailment_t = 0
            if action == 2 and power_action_MW < 0: # Wants to Discharge
                discharge_amount = min(abs(power_action_MW), self.soc)
                self.soc -= discharge_amount
                injection_t = min(G_t + discharge_amount, self.limit) 
            else:
                injection_t = min(G_t, self.limit)

        # Reward Calculation
        alpha, beta, gamma = 100.0, 1.0, 100000.0
        
        reward = - (alpha * curtailment_t)
        reward -= (beta * abs(power_action_MW))
        
        if injection_t > self.limit + 0.01: 
             reward -= gamma
        
        # Logging for non-training mode
        if not self.training_mode:
             # Note: Direct DataFrame assignment is slow, but kept for compatibility with original design logic
             # In a pure production env, we'd log to a list and create DF at the end.
             pass 

        self.current_step += 1
        self.done = (self.current_step >= len(self.df))
        
        next_state = self._get_state()
        return next_state, reward, self.done, {}
