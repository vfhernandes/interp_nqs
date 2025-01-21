import os
import numpy as np
import json
import netket as nk
from models import *  # Assuming models are imported from the `models` module
from utils import *  # Assuming utils are imported from the `utils` module
import flax
import sys

# Configurations and Patjs
index = sys.argv[1]
config_file = f"config_{index}.json"
script_dir = os.path.dirname(os.path.abspath(__file__))
config_folder = os.path.join(script_dir, 'configs')
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import netket.nn as nknn
import flax.linen as nn

import jax.numpy as jnp

class FFNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=2*x.shape[-1], 
                     use_bias=True, 
                     param_dtype=np.complex128, 
                     kernel_init=nn.initializers.normal(stddev=0.01), 
                     bias_init=nn.initializers.normal(stddev=0.01)
                    )(x)
        x = nknn.log_cosh(x)
        x = jnp.sum(x, axis=-1)
        return x
        

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
dj = config['dj']
j2j1_low = config['j2j1_low']
j2j1_high = config['j2j1_high']
L = system_size

baseline_js = [0.0, 1.0]

# Build js Array
js = np.arange(j2j1_low, j2j1_high + 1e-10, dj)
# Directory Structure
results_base_dir = os.path.join(script_dir, 'results', index)
os.makedirs(os.path.join(results_base_dir, 'baselines'), exist_ok=True)
os.makedirs(os.path.join(results_base_dir, 'finetune'), exist_ok=True)

# Function Definitions
def ensure_dir(path):
    """Ensures a directory exists."""
    os.makedirs(path, exist_ok=True)

def save_results(results, results_dir, j_i):
    """Saves results and variational state to disk."""
    # Save results as JSON
    results_path = os.path.join(results_dir, f'results_h{j_i:.3f}.json')
    with open(results_path, 'w') as file:
        json.dump(results, file, indent=4)

def save_vstate(vstate, results_dir, j_i):
    # Save vstate as binary
    vstate_path = os.path.join(results_dir, f'vstate_h{j_i:.3f}.mpack')
    with open(vstate_path, 'wb') as file:
        file.write(flax.serialization.to_bytes(vstate))

def random_vstate():
    # Define custom graph
    edge_colors = []
    for i in range(L):
        edge_colors.append([i, (i+1)%L, 1])
        edge_colors.append([i, (i+2)%L, 2])
    g = nk.graph.Graph(edges=edge_colors)
    hilbert = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=g.n_nodes)
    # model = symmetricRBM(alpha=alpha) if symmetric else nk.models.RBM(alpha=alpha)
    model = FFNN()
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

def train(j_i, vstate=None):
    """Train the model for a given h value."""
    results_for_j = {}  # A dictionary for results specific to this j_i
    
    # Define custom graph
    edge_colors = []
    J = [j_i, 1]
    
    for i in range(L):
        edge_colors.append([i, (i+1)%L, 1])
        edge_colors.append([i, (i+2)%L, 2])

    # Define the netket graph object
    g = nk.graph.Graph(edges=edge_colors)
    #Sigma^z*Sigma^z interactions
    sigmaz = [[1, 0], [0, -1]]
    mszsz = (np.kron(sigmaz, sigmaz))

    #Exchange interactions
    exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])

    bond_operator = [
        (J[0] * mszsz).tolist(),
        (J[1] * mszsz).tolist(),
        (-J[0] * exchange).tolist(),  
        (J[1] * exchange).tolist(),
    ]

    bond_color = [1, 2, 1, 2]
    
    hilbert = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=g.n_nodes)

    op = nk.operator.GraphOperator(hilbert, graph=g, bond_ops=bond_operator, bond_ops_colors=bond_color)

    # We shall use an exchange Sampler which preserves the global magnetization (as this is a conserved quantity in the model)
    sa = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=g, d_max = 2)
    # model = symmetricRBM(alpha=alpha) if symmetric else nk.models.RBM(alpha=alpha)
    model = FFNN()

    # Construct the variational state
    vs = nk.vqs.FullSumState(hilbert, model)

    # We choose a basic, albeit important, Optimizer: the Stochastic Gradient Descent
    opt = nk.optimizer.Sgd(learning_rate=0.01)

    # Stochastic Reconfiguration
    sr = nk.optimizer.SR(diag_shift=0.01, holomorphic = True)

    # We can then specify a Variational Monte Carlo object, using the Hamiltonian, sampler and optimizers chosen.
    # Note that we also specify the method to learn the parameters of the wave-function: here we choose the efficient
    # Stochastic reconfiguration (Sr), here in an iterative setup
    gs = nk.VMC(hamiltonian=op, optimizer=opt, variational_state=vs, preconditioner=sr)

    # We need to specify the local operators as a matrix acting on a local Hilbert space 
    sf = []
    sites = []
    structure_factor = nk.operator.LocalOperator(hilbert, dtype=complex)
    for i in range(0, L):
        for j in range(0, L):
            structure_factor += (nk.operator.spin.sigmaz(hilbert, i)*nk.operator.spin.sigmaz(hilbert, j))*((-1)**(i-j))/L

    E_gs, ket_gs = nk.exact.lanczos_ed(op, compute_eigenvectors=True)
    structure_factor_gs = (ket_gs.T.conj()@structure_factor.to_linear_operator()@ket_gs).real[0,0]
    exact_energy, exact_wavefunc = E_gs, ket_gs.reshape(-1)

    # Model Selection
    # model = symmetricRBM(alpha=alpha) if symmetric else nk.models.RBM(alpha=alpha)
    model = FFNN()
    if vstate is None:
        vstate = nk.vqs.FullSumState(hilbert, model)

    # Optimization and Training
    log = nk.logging.RuntimeLog()
    gs.run(out=log, n_iter=training_steps, obs={'Structure Factor': structure_factor}, show_progress = False)

    # Extract Results
    energy = log['Energy']['Mean']
    variance = log['Energy']['Variance']
    wavefunc = get_psi(vstate, hilbert, model)
    sf = np.real(log['Structure Factor']['Mean'])
    
    results_for_j['infidelity'] = infidelity(np.real(wavefunc), exact_wavefunc).item()
    results_for_j['wavefunc_real'] = np.real(wavefunc).tolist()
    results_for_j['wavefunc_imag'] = np.imag(wavefunc).tolist()
    results_for_j['sf'] = np.array(sf).tolist()
    results_for_j['sf_exact'] = structure_factor_gs.tolist()
    results_for_j['wavefunc_exact'] = exact_wavefunc.tolist()
    results_for_j['energy'] = np.real(energy).tolist()
    results_for_j['variance'] = np.array(variance).tolist()
    results_for_j['energy_exact'] = exact_energy.item()
    results_for_j['params_imag'] = jnp.imag(get_weights(vstate)).tolist()
    results_for_j['params_real'] = jnp.real(get_weights(vstate)).tolist()
    # results_for_j['activations'] = [
    #     hidden_activations(vstate.parameters, hilbert.all_states()[i], symmetric=symmetric).tolist()
    #     for i in range(2 ** system_size)
    # ]
    
    return vstate, results_for_j

def train_baseline():
    """Train for all baseline values of h."""
    baseline_results = {}
    baseline_dir = os.path.join(results_base_dir, 'baselines')
    
    for j_i in js:
        j_i = j_i.item()
        vstate, results = train(j_i)
        energy_error = np.abs(np.abs(results['energy']) - np.abs(results['energy_exact']))
        sf_error = np.abs(np.abs(results['sf']) - np.abs(results['sf_exact']))      
        infidelity = np.array(results['infidelity'])
        print(f"Baseline j2/j1={j_i:.3f}",
          ' | energy error: ', "{:.3e}".format(energy_error[-1].item()),
          # ' | avg wavefunc error: ', "{:.3e}".format(np.mean(wavefunc_error)), 
          ' | infidelity: ', "{:.3e}".format(infidelity),  
          ' | sf error: ', "{:.3e}".format(sf_error[-1].item()))
                #save_results(results, vstate, baseline_dir, j_i)
        save_vstate(vstate, baseline_dir, j_i)  
        baseline_results[f'{j_i:.3f}'] = results.copy()  # Store the results for each j_i
        
    return baseline_results

def fine_tune_with_single_baseline(single_baseline, baseline_dir, fine_tune_dir):
    """Fine-tune from a single baseline."""
    os.makedirs(os.path.join(fine_tune_dir, f'baseline_h{single_baseline:.3f}'), exist_ok=True)
    fine_tune_dir = os.path.join(fine_tune_dir, f'baseline_h{single_baseline:.3f}')
    vstate = load_vstate(os.path.join(baseline_dir, f'vstate_h{single_baseline:.3f}.mpack'), random_vstate())
    fine_tune_results = {}  # To accumulate fine-tune results for each j_i

    # Save baseline results in fine-tune directory
    baseline_results_path = os.path.join(baseline_dir, f'all_results.json')
    with open(baseline_results_path, 'r') as file:
        baseline_results = json.load(file)
    fine_tune_results[f'{single_baseline:.3f}'] = baseline_results[f'{single_baseline:.3f}']
    # save_results(baseline_results, fine_tune_dir, single_baseline)

    # Fine-tuning below baseline
    j_i = single_baseline - dj
    while j_i > j2j1_low - 1e-10:
        j_i = abs(j_i)
        vstate, results = train(j_i, vstate)

        energy_error = np.abs(np.abs(results['energy']) - np.abs(results['energy_exact']))
        wavefunc_error = np.abs(np.abs(results['wavefunc_real']) - np.abs(results['wavefunc_exact']))
        sf_error = np.abs(np.abs(results['sf']) - np.abs(results['sf_exact']))      
        infidelity = np.array(results['infidelity'])
        print(f"Fine-Tuning (below): Baseline j2/j1={single_baseline:.3f}, Current j2/j1={j_i:.3f}",
          ' | energy error: ', "{:.3e}".format(energy_error[-1].item()),
          #' | avg wavefunc error: ', "{:.3e}".format(np.mean(wavefunc_error)), 
          ' | infidelity: ', "{:.3e}".format(infidelity), 
          ' | sf error: ', "{:.3e}".format(sf_error[-1].item()))

        save_vstate(vstate, fine_tune_dir, j_i)
        fine_tune_results[f'{j_i:.3f}'] = results  # Store results for each j_i
        j_i -= dj

    # Reload baseline state for above training
    vstate = load_vstate(os.path.join(baseline_dir, f'vstate_h{single_baseline:.3f}.mpack'), vstate)

    # Fine-tuning above baseline
    j_i = single_baseline + dj
    while j_i < j2j1_high + 1e-10:
        vstate, results = train(j_i, vstate)

        energy_error = np.abs(np.abs(results['energy']) - np.abs(results['energy_exact']))
        wavefunc_error = np.abs(np.abs(results['wavefunc_real']) - np.abs(results['wavefunc_exact']))
        infidelity = np.array(results['infidelity'])
        sf_error = np.abs(np.abs(results['sf']) - np.abs(results['sf_exact']))      

        print(f"Fine-Tuning (below): Baseline j2/j1={single_baseline:.3f}, Current j2/j1={j_i:.3f}",
          ' | energy error: ', "{:.3e}".format(energy_error[-1].item()),
          #' | avg wavefunc error: ', "{:.3e}".format(np.mean(wavefunc_error)), 
          ' | infidelity: ', "{:.3e}".format(infidelity),
          ' | sf error: ', "{:.3e}".format(sf_error[-1].item()))

        save_vstate(vstate, fine_tune_dir, j_i) 
        fine_tune_results[f'{j_i:.3f}'] = results  # Store results for each j_i
        j_i += dj

    # Save the fine-tuning results for all j_i values
    save_all_results(fine_tune_results, fine_tune_dir)

def fine_tune_from_all_baselines(baseline_dir, fine_tune_dir):
    """Fine-tune from all baselines."""
    fine_tune_results_all = {}  # To store all fine-tune results for all baselines
    
    for j2j1_baseline in js:
        j2j1_baseline = j2j1_baseline.item()
        fine_tune_dir_single = os.path.join(fine_tune_dir, f'baseline_h{j2j1_baseline:.3f}')
        print(f"Starting fine-tuning for baseline j2/j1={j2j1_baseline:.3f}")
        fine_tune_results = fine_tune_with_single_baseline(j2j1_baseline, baseline_dir, fine_tune_dir_single)
        fine_tune_results_all[f'{j2j1_baseline:.3f}'] = fine_tune_results  # Store fine-tuning results for each baseline
    
    # Save all fine-tuning results
    save_all_results(fine_tune_results_all, fine_tune_dir)

# Baseline:
baseline_results = train_baseline()
save_all_results(baseline_results, os.path.join(results_base_dir, 'baselines'))


for j_base in baseline_js:
    if j_base not in js:
        print(f"Skipping j2/j1={j_base}, not in js array")
        continue
    # Fine-tune from a single baseline
    fine_tune_dir_single = os.path.join(results_base_dir, 'finetune')
    fine_tune_with_single_baseline(single_baseline=j_base, baseline_dir=os.path.join(results_base_dir, 'baselines'), fine_tune_dir=fine_tune_dir_single)


# # Fine-tune from all baselines
# fine_tune_dir_all = os.path.join(results_base_dir, 'finetune')
# fine_tune_from_all_baselines(baseline_dir=os.path.join(results_base_dir, 'baselines'), fine_tune_dir=fine_tune_dir_all)