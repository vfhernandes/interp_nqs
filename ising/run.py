import os
import numpy as np
import json
import netket as nk
from models import *  # Assuming models are imported from the `models` module
from utils import *  # Assuming utils are imported from the `utils` module
import flax
import sys

# Configurations and Paths
index = sys.argv[1]
config_file = "config_0.json"
script_dir = os.path.dirname(os.path.abspath(__file__))
config_folder = os.path.join(script_dir, 'configs')
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Load Configuration
config_path = os.path.join(config_folder, config_file)
with open(config_path, 'r') as file:
    config = json.load(file)

# Map Config Values to Variables
optimizer = config['optimizers']
optimizers = {'sgd': nk.optimizer.Sgd, 'adam': nk.optimizer.Adam}
learning_rate = config['learning_rates']
system_size = config['system_size']
alpha = config['alpha']
training_steps = config['training_steps']
symmetric = config['symmetric']
dh = config['dh']
h_low = config['h_low']
h_high = config['h_high']

# Override Configurations
"""
system_size = 4
alpha = 3
training_steps = 200
dh = 0.1
optimizer = 'adam'
learning_rate = 0.01
symmetric = False
"""

baseline_hs = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

# Build hs Array
hs = np.arange(h_low, h_high + 1e-10, dh)
# Directory Structure
results_base_dir = os.path.join(script_dir, 'results', index)
os.makedirs(os.path.join(results_base_dir, 'baselines'), exist_ok=True)
os.makedirs(os.path.join(results_base_dir, 'finetune'), exist_ok=True)

# Function Definitions
def ensure_dir(path):
    """Ensures a directory exists."""
    os.makedirs(path, exist_ok=True)

def save_results(results, results_dir, h_i):
    """Saves results and variational state to disk."""
    # Save results as JSON
    results_path = os.path.join(results_dir, f'results_h{h_i:.3f}.json')
    with open(results_path, 'w') as file:
        json.dump(results, file, indent=4)

def save_vstate(vstate, results_dir, h_i):
    # Save vstate as binary
    vstate_path = os.path.join(results_dir, f'vstate_h{h_i:.3f}.mpack')
    with open(vstate_path, 'wb') as file:
        file.write(flax.serialization.to_bytes(vstate))

def random_vstate():
    g = nk.graph.Chain(length=system_size, pbc=False)
    hilbert = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    model = symmetricRBM(alpha=alpha) if symmetric else nk.models.RBM(alpha=alpha)
    return nk.vqs.FullSumState(hilbert, model)


def load_vstate(path, vstate):
    """Loads a variational state from disk."""
    with open(path, 'rb') as file:
        return flax.serialization.from_bytes(vstate, file.read())

def save_all_results(results, results_dir):
    """Saves all results in a single JSON file."""
    results_path = os.path.join(results_dir, 'all_results.json')
    with open(results_path, 'w') as file:
        json.dump(results, file, indent=4)

def train(h_i, vstate=None):
    """Train the model for a given h value."""
    results_for_h = {}  # A dictionary for results specific to this h_i
    J = -1  
    g = nk.graph.Chain(length=system_size, pbc=False)
    hilbert = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    hamiltonian = nk.operator.Ising(hilbert, h=h_i, graph=g)
    wv = nk.exact.lanczos_ed(hamiltonian, k=1, compute_eigenvectors=True)
    exact_energy, exact_wavefunc = wv[0][0], wv[1].reshape(-1)

    # Model Selection
    model = symmetricRBM(alpha=alpha) if symmetric else nk.models.RBM(alpha=alpha)
    if vstate is None:
        vstate = nk.vqs.FullSumState(hilbert, model)

    # Optimization and Training
    optim = optimizers[optimizer](learning_rate=learning_rate)
    gs = nk.driver.VMC(hamiltonian, optim, variational_state=vstate, preconditioner=nk.optimizer.SR(diag_shift=0.1))
    log = nk.logging.RuntimeLog()
    gs.run(n_iter=training_steps, out=log, show_progress=False)

    # Extract Results
    energy = log['Energy']['Mean']
    variance = log['Energy']['Variance']
    wavefunc = get_psi(vstate, hilbert, model)
    
    results_for_h['infidelity'] = infidelity(wavefunc, exact_wavefunc).item()
    results_for_h['wavefunc'] = np.array(wavefunc).tolist()
    results_for_h['wavefunc_exact'] = exact_wavefunc.tolist()
    results_for_h['energy'] = np.array(energy).tolist()
    results_for_h['variance'] = np.array(variance).tolist()
    results_for_h['energy_exact'] = exact_energy.item()
    results_for_h['params'] = np.array(get_weights(vstate, symmetric=symmetric)).tolist()
    results_for_h['activations'] = [
        hidden_activations(vstate.parameters, hilbert.all_states()[i], symmetric=symmetric).tolist()
        for i in range(2 ** system_size)
    ]
    
    return vstate, results_for_h

def train_baseline():
    """Train for all baseline values of h."""
    baseline_results = {}
    baseline_dir = os.path.join(results_base_dir, 'baselines')
    
    for h_i in hs:
        h_i = h_i.item()
        vstate, results = train(h_i)
        energy_error = np.abs(np.abs(results['energy']) - np.abs(results['energy_exact']))
        wavefunc_error = np.abs(np.abs(results['wavefunc']) - np.abs(results['wavefunc_exact']))
        infidelity = np.array(results['infidelity'])
        print(f"Baseline h={h_i:.3f}",
          ' | energy error: ', "{:.3e}".format(energy_error[-1].item()),
          # ' | avg wavefunc error: ', "{:.3e}".format(np.mean(wavefunc_error)), 
          ' | infidelity: ', "{:.3e}".format(infidelity))    
                #save_results(results, vstate, baseline_dir, h_i)
        save_vstate(vstate, baseline_dir, h_i)  
        baseline_results[f'{h_i:.3f}'] = results.copy()  # Store the results for each h_i
        
    return baseline_results

def fine_tune_with_single_baseline(single_baseline, baseline_dir, fine_tune_dir):
    """Fine-tune from a single baseline."""
    os.makedirs(os.path.join(fine_tune_dir, f'baseline_h{single_baseline:.3f}'), exist_ok=True)
    fine_tune_dir = os.path.join(fine_tune_dir, f'baseline_h{single_baseline:.3f}')
    vstate = load_vstate(os.path.join(baseline_dir, f'vstate_h{single_baseline:.3f}.mpack'), random_vstate())
    fine_tune_results = {}  # To accumulate fine-tune results for each h_i

    # Save baseline results in fine-tune directory
    baseline_results_path = os.path.join(baseline_dir, f'all_results.json')
    with open(baseline_results_path, 'r') as file:
        baseline_results = json.load(file)
    fine_tune_results[f'{single_baseline:.3f}'] = baseline_results[f'{single_baseline:.3f}']
    # save_results(baseline_results, fine_tune_dir, single_baseline)

    # Fine-tuning below baseline
    h_i = single_baseline - dh
    while h_i > h_low - 1e-10:
        h_i = abs(h_i)
        vstate, results = train(h_i, vstate)

        energy_error = np.abs(np.abs(results['energy']) - np.abs(results['energy_exact']))
        wavefunc_error = np.abs(np.abs(results['wavefunc']) - np.abs(results['wavefunc_exact']))
        infidelity = np.array(results['infidelity'])
        print(f"Fine-Tuning (below): Baseline h={single_baseline:.3f}, Current h={h_i:.3f}",
          ' | energy error: ', "{:.3e}".format(energy_error[-1].item()),
          #' | avg wavefunc error: ', "{:.3e}".format(np.mean(wavefunc_error)), 
          ' | infidelity: ', "{:.3e}".format(infidelity))    
        
        save_vstate(vstate, fine_tune_dir, h_i)
        fine_tune_results[f'{h_i:.3f}'] = results  # Store results for each h_i
        h_i -= dh

    # Reload baseline state for above training
    vstate = load_vstate(os.path.join(baseline_dir, f'vstate_h{single_baseline:.3f}.mpack'), vstate)

    # Fine-tuning above baseline
    h_i = single_baseline + dh
    while h_i < h_high + 1e-10:
        vstate, results = train(h_i, vstate)

        energy_error = np.abs(np.abs(results['energy']) - np.abs(results['energy_exact']))
        wavefunc_error = np.abs(np.abs(results['wavefunc']) - np.abs(results['wavefunc_exact']))
        infidelity = np.array(results['infidelity'])

        print(f"Fine-Tuning (below): Baseline h={single_baseline:.3f}, Current h={h_i:.3f}",
          ' | energy error: ', "{:.3e}".format(energy_error[-1].item()),
          #' | avg wavefunc error: ', "{:.3e}".format(np.mean(wavefunc_error)), 
          ' | infidelity: ', "{:.3e}".format(infidelity))    
        
        save_vstate(vstate, fine_tune_dir, h_i) 
        fine_tune_results[f'{h_i:.3f}'] = results  # Store results for each h_i
        h_i += dh

    # Save the fine-tuning results for all h_i values
    save_all_results(fine_tune_results, fine_tune_dir)

def fine_tune_from_all_baselines(baseline_dir, fine_tune_dir):
    """Fine-tune from all baselines."""
    fine_tune_results_all = {}  # To store all fine-tune results for all baselines
    
    for h_baseline in hs:
        h_baseline = h_baseline.item()
        fine_tune_dir_single = os.path.join(fine_tune_dir, f'baseline_h{h_baseline:.3f}')
        print(f"Starting fine-tuning for baseline h={h_baseline:.3f}")
        fine_tune_results = fine_tune_with_single_baseline(h_baseline, baseline_dir, fine_tune_dir_single)
        fine_tune_results_all[f'{h_baseline:.3f}'] = fine_tune_results  # Store fine-tuning results for each baseline
    
    # Save all fine-tuning results
    save_all_results(fine_tune_results_all, fine_tune_dir)

# Baseline:
baseline_results = train_baseline()
save_all_results(baseline_results, os.path.join(results_base_dir, 'baselines'))


for h_base in baseline_hs:
    if h_base not in hs:
        print(f"Skipping h={h_base}, not in hs array")
        continue
    # Fine-tune from a single baseline
    fine_tune_dir_single = os.path.join(results_base_dir, 'finetune')
    fine_tune_with_single_baseline(single_baseline=h_base, baseline_dir=os.path.join(results_base_dir, 'baselines'), fine_tune_dir=fine_tune_dir_single)


# # Fine-tune from all baselines
# fine_tune_dir_all = os.path.join(results_base_dir, 'finetune')
# fine_tune_from_all_baselines(baseline_dir=os.path.join(results_base_dir, 'baselines'), fine_tune_dir=fine_tune_dir_all)