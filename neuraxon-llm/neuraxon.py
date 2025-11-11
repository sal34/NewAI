"""
Neuraxon: Bio-inspired Neural Network with Trinary States
Based on the paper "Neuraxon"
Hibridized with Aigarth Intelligent Tissue https://github.com/Aigarth/aigarth-it

This implementation includes:
- Trinary neuron states (-1, 0, 1)
- Ring architecture (input, hidden, output neurons)
- Multiple synapse types (ionotropic fast/slow, metabotropic)
- Neuromodulators (dopamine, serotonin, acetylcholine, norepinephrine)
- Synaptic plasticity (growth, death, silent synapses)
- Spontaneous neural activity
- Continuous processing model
"""

import json
import random
import math
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from tokenizer import Tokenizer
from neuraxon_cython import update_neurons_cython, update_synapses_cython


# =============================================================================
# NETWORK PARAMETERS - Biologically Plausible Ranges
# =============================================================================

@dataclass
class NetworkParameters:
    """Default network parameters with biologically plausible ranges"""

    # Network Architecture
    num_input_neurons: int = 5         # Range: [1, 100]
    num_hidden_neurons: int = 20      # Range: [1, 1000]
    num_output_neurons: int = 5      # Range: [1, 100]
    connection_probability: float = 0.05  # Range: [0.0, 1.0]

    # Neuron Parameters
    membrane_time_constant: float = 20.0      # ms, Range: [5.0, 50.0]
    firing_threshold_excitatory: float = 1.0  # Range: [0.5, 2.0]
    firing_threshold_inhibitory: float = -1.0 # Range: [-2.0, -0.5]
    adaptation_rate: float = 0.05             # Range: [0.0, 0.2]
    spontaneous_firing_rate: float = 0.01     # Range: [0.0, 0.1]
    neuron_health_decay: float = 0.0001        # Range: [0.0, 0.01]

    # Synapse Parameters - Fast (Ionotropic)
    tau_fast: float = 5.0              # ms, Range: [1.0, 10.0]
    w_fast_init_min: float = -1.0     # Range: [-1.0, 0.0]
    w_fast_init_max: float = 1.0      # Range: [0.0, 1.0]

    # Synapse Parameters - Slow (NMDA-like)
    tau_slow: float = 50.0             # ms, Range: [20.0, 100.0]
    w_slow_init_min: float = -0.5     # Range: [-1.0, 0.0]
    w_slow_init_max: float = 0.5      # Range: [0.0, 1.0]

    # Synapse Parameters - Metabotropic
    tau_meta: float = 1000.0           # ms, Range: [500.0, 5000.0]
    w_meta_init_min: float = -0.3     # Range: [-0.5, 0.0]
    w_meta_init_max: float = 0.3      # Range: [0.0, 0.5]

    # Plasticity Parameters
    learning_rate: float = 0.01        # Range: [0.0, 0.1]
    stdp_window: float = 20.0         # ms, Range: [10.0, 50.0]
    synapse_integrity_threshold: float = 0.1  # Range: [0.0, 0.5]
    synapse_formation_prob: float = 0.05      # Range: [0.0, 0.2]
    synapse_death_prob: float = 0.001          # Range: [0.0, 0.1]
    neuron_death_threshold: float = 0.1       # Range: [0.0, 0.3]

    # Neuromodulator Parameters
    dopamine_baseline: float = 0.1     # Range: [0.0, 1.0]
    serotonin_baseline: float = 0.1    # Range: [0.0, 1.0]
    acetylcholine_baseline: float = 0.1  # Range: [0.0, 1.0]
    norepinephrine_baseline: float = 0.1  # Range: [0.0, 1.0]
    neuromod_decay_rate: float = 0.1   # Range: [0.0, 0.5]

    # Simulation Parameters
    dt: float = 1.0                    # ms, Range: [0.1, 10.0]
    simulation_steps: int = 100        # Range: [1, 10000]


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class NeuronType(Enum):
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"


class SynapseType(Enum):
    IONOTROPIC_FAST = "ionotropic_fast"
    IONOTROPIC_SLOW = "ionotropic_slow"
    METABOTROPIC = "metabotropic"
    SILENT = "silent"


class TrinaryState(Enum):
    INHIBITORY = -1
    NEUTRAL = 0
    EXCITATORY = 1


# =============================================================================
# NEURAXON NETWORK CLASS
# =============================================================================

class NeuraxonNetwork:
    """
    Complete Neuraxon network with ring architecture
    """

    def __init__(self, params: Optional[NetworkParameters] = None):
        self.params = params or NetworkParameters()

        # Data structures for Numba
        self.neurons = None
        self.synapses = None
        self.input_neuron_indices = None
        self.hidden_neuron_indices = None
        self.output_neuron_indices = None

        # Neuromodulators (global state)
        self.neuromodulators = np.array([
            self.params.dopamine_baseline,
            self.params.serotonin_baseline,
            self.params.acetylcholine_baseline,
            self.params.norepinephrine_baseline
        ], dtype=np.float32)

        # Simulation state
        self.time = 0.0
        self.step_count = 0

        # Initialize network
        self._initialize_neurons()
        self._initialize_synapses()

    def _initialize_neurons(self):
        """Create neurons with ring architecture"""
        num_neurons = self.params.num_input_neurons + self.params.num_hidden_neurons + self.params.num_output_neurons
        # neuron state: [trinary_state, membrane_potential, adaptation, autoreceptor, health, type, is_hidden, is_active]
        self.neurons = np.zeros((num_neurons, 8), dtype=np.float32)

        self.input_neuron_indices = np.arange(self.params.num_input_neurons)
        self.hidden_neuron_indices = np.arange(self.params.num_input_neurons, self.params.num_input_neurons + self.params.num_hidden_neurons)
        self.output_neuron_indices = np.arange(self.params.num_input_neurons + self.params.num_hidden_neurons, num_neurons)

        self.neurons[self.input_neuron_indices, 5] = 0 # NeuronType.INPUT
        self.neurons[self.hidden_neuron_indices, 5] = 1 # NeuronType.HIDDEN
        self.neurons[self.output_neuron_indices, 5] = 2 # NeuronType.OUTPUT

        self.neurons[self.hidden_neuron_indices, 6] = 1 # is_hidden
        self.neurons[:, 7] = 1 # is_active
        self.neurons[:, 4] = 1.0 # health

    def _initialize_synapses(self):
        """
        Create synapses using the Watts-Strogatz small-world model.
        """
        num_neurons = len(self.neurons)
        k = max(4, int(2 * math.log(num_neurons)))
        if k % 2 != 0:
            k += 1

        p = self.params.connection_probability

        graph = nx.watts_strogatz_graph(num_neurons, k, p)

        num_synapses = len(graph.edges())
        # synapse state: [pre_id, post_id, w_fast, w_slow, w_meta, is_silent, is_modulatory, integrity, pre_trace, post_trace]
        self.synapses = np.zeros((num_synapses, 10), dtype=np.float32)

        for i, (pre_id, post_id) in enumerate(graph.edges()):
            self.synapses[i, 0] = pre_id
            self.synapses[i, 1] = post_id
            self.synapses[i, 2] = random.uniform(self.params.w_fast_init_min, self.params.w_fast_init_max)
            self.synapses[i, 3] = random.uniform(self.params.w_slow_init_min, self.params.w_slow_init_max)
            self.synapses[i, 4] = random.uniform(self.params.w_meta_init_min, self.params.w_meta_init_max)
            self.synapses[i, 5] = 1 if random.random() < 0.1 else 0
            self.synapses[i, 6] = 1 if random.random() < 0.2 else 0
            self.synapses[i, 7] = 1.0

    def simulate_step(self, external_inputs: Optional[Dict[int, float]] = None):
        """
        Simulate one time step

        Args:
            external_inputs: Dict mapping neuron IDs to external input values
        """
        external_inputs_array = np.zeros(len(self.neurons), dtype=np.float32)
        if external_inputs:
            for i, val in external_inputs.items():
                external_inputs_array[i] = val

        synaptic_inputs = np.zeros(len(self.neurons), dtype=np.float32)
        modulatory_inputs = np.zeros(len(self.neurons), dtype=np.float32)

        for i in range(len(self.synapses)):
            pre_id = int(self.synapses[i, 0])
            post_id = int(self.synapses[i, 1])
            if self.neurons[pre_id, 7]: # is_active
                syn_input = (self.synapses[i, 2] + self.synapses[i, 3]) * self.neurons[pre_id, 0]
                if not self.synapses[i, 5]: # is_silent
                    synaptic_inputs[post_id] += syn_input
                if self.synapses[i, 6]: # is_modulatory
                    modulatory_inputs[post_id] += self.synapses[i, 4]

        params_array = np.array(list(asdict(self.params).values()), dtype=np.float32)

        update_neurons_cython(
            self.neurons,
            synaptic_inputs,
            modulatory_inputs,
            external_inputs_array,
            self.neuromodulators,
            params_array,
            self.params.dt
        )

        update_synapses_cython(
            self.synapses,
            self.neurons,
            self.neuromodulators,
            params_array,
            self.params.dt
        )

        # Update neuromodulators (decay)
        baselines = np.array([
            self.params.dopamine_baseline,
            self.params.serotonin_baseline,
            self.params.acetylcholine_baseline,
            self.params.norepinephrine_baseline
        ], dtype=np.float32)
        self.neuromodulators += (
            (baselines - self.neuromodulators) *
            self.params.neuromod_decay_rate * self.params.dt / 100.0
        )

        # Structural plasticity
        self._apply_structural_plasticity()

        # Update time
        self.time += self.params.dt
        self.step_count += 1

    def _apply_structural_plasticity(self):
        """Apply synapse formation and death based on more nuanced conditions."""
        # Synapse pruning based on integrity
        self.synapses = self.synapses[self.synapses[:, 7] > self.params.synapse_integrity_threshold]

        # Neuron death for unhealthy hidden neurons
        for i in self.hidden_neuron_indices:
            if self.neurons[i, 4] < self.params.neuron_death_threshold and random.random() < 0.01:
                self.neurons[i, 7] = 0
                self.synapses = self.synapses[(self.synapses[:, 0] != i) & (self.synapses[:, 1] != i)]

        # Synapse formation between healthy and active neurons
        if random.random() < self.params.synapse_formation_prob:
            healthy_neurons = np.where((self.neurons[:, 7] == 1) & (self.neurons[:, 4] > 0.5))[0]
            if len(healthy_neurons) >= 2:
                pre = random.choice(healthy_neurons)
                post = random.choice(healthy_neurons)

                if pre != post and not (self.neurons[pre, 5] == 2 and self.neurons[post, 5] == 0):
                    exists = np.any((self.synapses[:, 0] == pre) & (self.synapses[:, 1] == post))
                    if not exists:
                        new_synapse = np.zeros((1, 10), dtype=np.float32)
                        new_synapse[0, 0] = pre
                        new_synapse[0, 1] = post
                        new_synapse[0, 2] = random.uniform(self.params.w_fast_init_min, self.params.w_fast_init_max)
                        new_synapse[0, 3] = random.uniform(self.params.w_slow_init_min, self.params.w_slow_init_max)
                        new_synapse[0, 4] = random.uniform(self.params.w_meta_init_min, self.params.w_meta_init_max)
                        new_synapse[0, 5] = 1 if random.random() < 0.1 else 0
                        new_synapse[0, 6] = 1 if random.random() < 0.2 else 0
                        new_synapse[0, 7] = 1.0
                        self.synapses = np.vstack([self.synapses, new_synapse])

    def set_input_states(self, states: List[int]):
        """
        Set states of input neurons

        Args:
            states: List of trinary states (-1, 0, 1)
        """
        for i, state in enumerate(states[:len(self.input_neuron_indices)]):
            self.neurons[self.input_neuron_indices[i], 0] = state
            self.neurons[self.input_neuron_indices[i], 1] = state * self.params.firing_threshold_excitatory

    def get_output_states(self) -> List[int]:
        """Get current states of output neurons"""
        return self.neurons[self.output_neuron_indices, 0][self.neurons[self.output_neuron_indices, 7] == 1].tolist()

    def modulate(self, neuromodulator: str, level: float):
        """
        Adjust neuromodulator level

        Args:
            neuromodulator: Name of neuromodulator
            level: New level (0.0 to 1.0)
        """
        if neuromodulator == 'dopamine':
            self.neuromodulators[0] = max(0.0, min(1.0, level))
        elif neuromodulator == 'serotonin':
            self.neuromodulators[1] = max(0.0, min(1.0, level))
        elif neuromodulator == 'acetylcholine':
            self.neuromodulators[2] = max(0.0, min(1.0, level))
        elif neuromodulator == 'norepinephrine':
            self.neuromodulators[3] = max(0.0, min(1.0, level))

    def to_dict(self) -> dict:
        """Convert network to dictionary for JSON serialization"""
        return {
            'parameters': asdict(self.params),
            'neurons': self.neurons.tolist(),
            'synapses': self.synapses.tolist(),
            'neuromodulators': self.neuromodulators.tolist(),
            'time': self.time,
            'step_count': self.step_count
        }


# =============================================================================
# JSON SAVE/LOAD FUNCTIONS
# =============================================================================

def save_network(network: NeuraxonNetwork, filename: str):
    """
    Save network to JSON file

    Args:
        network: NeuraxonNetwork instance
        filename: Output filename
    """
    with open(filename, 'w') as f:
        json.dump(network.to_dict(), f, indent=2)
    print(f"Network saved to {filename}")


def load_network(filename: str) -> NeuraxonNetwork:
    """
    Load network from JSON file

    Args:
        filename: Input filename

    Returns:
        Reconstructed NeuraxonNetwork
    """
    with open(filename, 'r') as f:
        data = json.load(f)

    # Reconstruct parameters
    params = NetworkParameters(**data['parameters'])

    # Create network
    network = NeuraxonNetwork(params)

    # Restore network state
    network.neurons = np.array(data['neurons'], dtype=np.float32)
    synapses_data = data.get('synapses', [])
    if not synapses_data:
        network.synapses = np.empty((0, 10), dtype=np.float32)
    else:
        network.synapses = np.array(synapses_data, dtype=np.float32)
    network.neuromodulators = np.array(data['neuromodulators'], dtype=np.float32)
    network.time = data['time']
    network.step_count = data['step_count']

    print(f"Network loaded from {filename}")
    return network


# =============================================================================
# LANGUAGE MODEL CLASS
# =============================================================================

class LanguageModel(NeuraxonNetwork):
    """
    A language model based on the Neuraxon network.
    """
    def __init__(self, tokenizer: Tokenizer, params: NetworkParameters = None):
        """
        Initializes the language model.

        Args:
            tokenizer: A tokenizer object.
            params: Network parameters.
        """
        self.tokenizer = tokenizer

        # Adjust network parameters for language modeling
        if params is None:
            params = NetworkParameters()

        params.num_input_neurons = tokenizer.vocab_size
        params.num_output_neurons = tokenizer.vocab_size

        super().__init__(params)

    @classmethod
    def load(cls, filepath: str, tokenizer: Tokenizer):
        """
        Loads a language model from a file.

        Args:
            filepath: The path to the model file.
            tokenizer: The tokenizer to use with the model.

        Returns:
            A LanguageModel instance.
        """
        network = load_network(filepath)
        model = cls(tokenizer, network.params)
        model.neurons = network.neurons
        model.synapses = network.synapses
        model.neuromodulators = network.neuromodulators
        model.time = network.time
        model.step_count = network.step_count

        print(f"Language model loaded from {filepath}")
        return model

    def process_text(self, text: str, duration_per_char: int = 5):
        """
        Processes a string of text, feeding it into the network one character at a time
        with continuous time processing.
        """
        encoded_text = self.tokenizer.encode(text)

        for char_int in encoded_text:
            # Convert the character to a trinary vector
            input_vector = self.tokenizer.int_to_trinary_vector(char_int)

            # Set the input neuron states
            self.set_input_states(input_vector)

            # Simulate for a duration to allow the network to process the character
            for _ in range(duration_per_char):
                self.simulate_step()

    def predict_next_char(self, input_text: str) -> str:
        """
        Predicts the next character in a sequence.

        Args:
            input_text: The input text.

        Returns:
            The predicted next character.
        """
        # Process the input text to set the network's state
        self.process_text(input_text)

        # Get the output neuron states
        output_states = self.get_output_states()

        # Convert the output states to an integer
        predicted_int = self.tokenizer.trinary_vector_to_int(output_states)

        # Decode the integer to a character
        return self.tokenizer.decode([predicted_int])



if __name__ == "__main__":
    print("="*70)
    print("NEURAXON - Bio-Inspired Neural Network")
    print("="*70)

    # Create network with default parameters
    print("\n1. Creating network...")
    params = NetworkParameters()
    network = NeuraxonNetwork(params)

    print(f"   - Input neurons: {len(network.input_neuron_indices)}")
    print(f"   - Hidden neurons: {len(network.hidden_neuron_indices)}")
    print(f"   - Output neurons: {len(network.output_neuron_indices)}")
    print(f"   - Total synapses: {len(network.synapses)}")

    # Set input states
    print("\n2. Setting input states...")
    input_pattern = [1, -1, 0, 1, -1]
    network.set_input_states(input_pattern)
    print(f"   Input pattern: {input_pattern}")

    # Run simulation
    print("\n3. Running simulation...")
    for step in range(10):
        network.simulate_step()

        if step % 20 == 0:
            outputs = network.get_output_states()
            print(f"   Step {step}: Outputs = {outputs}")

    # Modulate network
    print("\n4. Testing neuromodulation...")
    network.modulate('dopamine', 0.8)
    print(f"   Dopamine level set to 0.8")

    # Save network
    print("\n5. Saving network...")
    save_network(network, "neuraxon_network.json")
