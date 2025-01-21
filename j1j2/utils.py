import netket as nk
import jax.numpy as jnp
import numpy as np


def get_psi(vstate,hi,model):
    '''
    Function that returns a vector with the wavefunction probability amplitudes

    Inputs ---
    vsate: Netket's vstate object, the variational quantum state
    hi:    Hilbert space 
    model: Instance of RBM model

    Outputs ---
    psi:   Vector of probability amplitudes indexed all possible by configurations/eigenstates of the Hilbert space

    '''
    parameters = vstate.variables
    all_configurations = hi.all_states()

    logpsi = model.apply(parameters,all_configurations)
    psi = jnp.exp(logpsi)
    psi = psi/ jnp.linalg.norm(psi)

    return psi

    
def hidden_activations(params, v, symmetric = False):
    """
    Compute the activations of the hidden units given the visible units using log(cosh).
    h_probs = log(cosh(v @ W + c))
    """
    if symmetric:
        W, c = params['RBM_0']['Dense']['kernel'], params['RBM_0']['Dense']['bias']
    else:
        W, c = params['Dense']['kernel'], params['Dense']['bias']
    pre_activation = v @ W + c  # Linear transformation
    h_probs = nk.nn.activation.log_cosh(pre_activation)  # Log-cosh activation
    return h_probs

def infidelity(wavefunc1, wavefunc2):

    norm = np.linalg.norm(wavefunc1)*np.linalg.norm(wavefunc2)
    overlap = np.dot(np.conjugate(wavefunc1), wavefunc2)
    fidelity = np.abs(overlap)**2/norm

    return 1 - fidelity

def get_weights(vstate):
    '''
    Function that returns flattened array containing the RBM weights

    Input:      Netket's vstate object, the variational quantum state. 
    Output:     Flattened 1D array with RBM weights
    '''

    params = vstate.parameters
    weights = params['Dense_0']['kernel'].reshape(-1).tolist()
    weights.extend(params['Dense_0']['bias'].tolist())
    return jnp.array(weights)