import json
import os

NOTEBOOK_PATH = "notebooks/02_APPR_DQN_Training.ipynb"

def update_notebook_torch():
    print(f"Reading {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Update Imports (Add Agent, Remove Tensorflow/Keras)
    # Actually, let's just make sure imports are clean.
    first_code_cell = next(c for c in nb['cells'] if c['cell_type'] == 'code')
    source = "".join(first_code_cell['source'])
    
    new_imports = [
        "import sys\n",
        "import os\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import random\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Add src to path\n",
        "sys.path.append(os.path.abspath('..'))\n",
        "from src.appr_core import GridEnvironment, enrich_data_with_forecast\n",
        "from src.appr_agent import DQNAgent # PyTorch Agent\n"
    ]
    
    first_code_cell['source'] = new_imports
    print("ACTION: Updated imports to include DQNAgent and remove TF.")

    # 2. Find and Replace the Training Loop / Agent Setup
    # Search for where 'env' is initialized, then subsequently where 'q_network' was built.
    
    # We will look for "PARTE 2: IMPLEMENTACIÓN DEL AGENTE DQN" or similar
    # And replace the manual Keras definition with DQNAgent usage.
    
    cells_to_keep = []
    
    skip_mode = False
    
    for cell in nb['cells']:
        src_text = "".join(cell['source'])
        
        # Detect start of Old Agent Section
        if "PARTE 2: IMPLEMENTACIÓN DEL AGENTE DQN" in src_text:
            cells_to_keep.append(cell) # Keep the header
            
            # Create a new cell for Agent Initialization
            new_code = [
                 "# --- INICIALIZACIÓN DEL AGENTE PYTORCH ---\n",
                 "state_size = env.state_space_size\n",
                 "action_size = env.action_space_size\n",
                 "\n",
                 "agent = DQNAgent(state_size, action_size)\n",
                 "print('Agente DQN (PyTorch) inicializado.')\n"
            ]
            cells_to_keep.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": new_code
            })
            
            # Start skipping until we hit Training Loop
            skip_mode = True
            print("ACTION: Replacing Agent Definition block.")
            continue
            
        if "Iniciando entrenamiento del APPR" in src_text:
             # Found training loop. We need to replace it or update it.
             # The old loop used 'q_network(...)'. The new one uses 'agent.act(...)'.
             # Let's replace the whole training cell with the new PyTorch loop.
             skip_mode = False
             
             new_loop = [
                 "# --- Bucle de Entrenamiento ---\n",
                 "EPISODES = 100\n",
                 "history = []\n",
                 "print(f\"\\nIniciando entrenamiento del APPR para {EPISODES} episodios...\")\n",
                 "\n",
                 "for e in range(EPISODES):\n",
                 "    state = env.reset()\n",
                 "    episode_reward = 0\n",
                 "    done = False\n",
                 "    \n",
                 "    while not done:\n",
                 "        action = agent.act(state)\n",
                 "        next_state, reward, done, _ = env.step(action)\n",
                 "        agent.remember(state, action, reward, next_state, done)\n",
                 "        agent.replay()\n",
                 "        state = next_state\n",
                 "        episode_reward += reward\n",
                 "        if done:\n",
                 "            break\n",
                 "\n",
                 "    agent.decay_epsilon()\n",
                 "    if e % 10 == 0:\n",
                 "        agent.update_target_network()\n",
                 "        print(f\"Episodio {e}/{EPISODES} | Recompensa Total: {episode_reward:.2f} | Epsilon: {agent.epsilon:.3f}\")\n",
                 "    history.append(episode_reward)\n"
             ]
             
             cells_to_keep.append({
                 "cell_type": "code",
                 "execution_count": None,
                 "metadata": {},
                 "outputs": [],
                 "source": new_loop
             })
             print("ACTION: Replaced Training Loop.")
             continue

        # Skip intermediate Keras defs
        if skip_mode:
            # Check if this cell is purely Keras code?
            if "build_dqn" in src_text or "q_network =" in src_text:
                continue
            # If we hit training loop (caught above), we stop skipping.
            # But what if there are other cells?
            # Safe bet: The original notebook had Agent Def -> Training Loop.
            pass

        cells_to_keep.append(cell)

    # 3. Evaluation Section
    # Need to update "q_values = q_network(...)" to "agent.act(..., training=False)"
    
    final_cells = []
    for cell in cells_to_keep:
        src_text = "".join(cell['source'])
        if "PARTE 3: EVALUACIÓN FINAL" in src_text or "q_network(" in src_text:
            # Update inference calls
            new_source = []
            for line in cell['source']:
                if "q_values = q_network" in line:
                    new_source.append("    # Usar agente para predicción\n")
                    new_source.append("    action = agent.act(state, training=False)\n")
                elif "action = np.argmax" in line:
                    pass # Handled above
                elif "env.training_mode = False" in line:
                    new_source.append(line)
                else:
                    new_source.append(line)
            cell['source'] = new_source
        final_cells.append(cell)

    nb['cells'] = final_cells
    
    print(f"Saving updated notebook to {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("DONE.")

if __name__ == "__main__":
    update_notebook_torch()
