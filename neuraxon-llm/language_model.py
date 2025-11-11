from neuraxon import NeuraxonNetwork, NetworkParameters
from tokenizer import Tokenizer
from typing import List

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

        # We can also adjust other parameters if needed, for example:
        # params.num_hidden_neurons = 128
        # params.connection_probability = 0.1

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
        import json
        from neuraxon import Synapse

        with open(filepath, 'r') as f:
            data = json.load(f)

        params = NetworkParameters(**data['parameters'])
        model = cls(tokenizer, params)

        # Restore neuron states
        for neuron_data in data['neurons']['input']:
            neuron = model.input_neurons[neuron_data['id'] % len(model.input_neurons)]
            neuron.membrane_potential = neuron_data['membrane_potential']
            neuron.trinary_state = neuron_data['trinary_state']
            neuron.health = neuron_data['health']
            neuron.is_active = neuron_data['is_active']

        for neuron_data in data['neurons']['hidden']:
            idx = neuron_data['id'] - len(model.input_neurons)
            if 0 <= idx < len(model.hidden_neurons):
                neuron = model.hidden_neurons[idx]
                neuron.membrane_potential = neuron_data['membrane_potential']
                neuron.trinary_state = neuron_data['trinary_state']
                neuron.health = neuron_data['health']
                neuron.is_active = neuron_data['is_active']

        for neuron_data in data['neurons']['output']:
            idx = neuron_data['id'] - len(model.input_neurons) - len(model.hidden_neurons)
            if 0 <= idx < len(model.output_neurons):
                neuron = model.output_neurons[idx]
                neuron.membrane_potential = neuron_data['membrane_potential']
                neuron.trinary_state = neuron_data['trinary_state']
                neuron.health = neuron_data['health']
                neuron.is_active = neuron_data['is_active']

        # Restore synapse states
        model.synapses = []
        for syn_data in data['synapses']:
            synapse = Synapse(syn_data['pre_id'], syn_data['post_id'], params)
            synapse.w_fast = syn_data['w_fast']
            synapse.w_slow = syn_data['w_slow']
            synapse.w_meta = syn_data['w_meta']
            synapse.is_silent = syn_data['is_silent']
            synapse.is_modulatory = syn_data['is_modulatory']
            synapse.integrity = syn_data['integrity']
            model.synapses.append(synapse)

        model.neuromodulators = data['neuromodulators']
        model.time = data['time']
        model.step_count = data['step_count']

        print(f"Language model loaded from {filepath}")
        return model

    def process_text(self, text: str):
        """
        Processes a string of text, feeding it into the network one character at a time.
        """
        encoded_text = self.tokenizer.encode(text)

        for char_int in encoded_text:
            # Convert the character to a trinary vector
            input_vector = self.tokenizer.int_to_trinary_vector(char_int)

            # Set the input neuron states
            self.set_input_states(input_vector)

            # Simulate one step
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

if __name__ == '__main__':
    # Example Usage
    from datasets import load_dataset

    # Load a sample corpus
    print("Loading dataset...")
    dataset = load_dataset("nampdn-ai/tiny-textbooks", split="train", streaming=True)
    sample_corpus = [item['text'] for item in dataset.take(10)]

    # Create a tokenizer
    print("Creating tokenizer...")
    tokenizer = Tokenizer(sample_corpus)

    # Create the language model
    print("Creating language model...")
    lm = LanguageModel(tokenizer)

    print(f"Input neurons: {lm.params.num_input_neurons}")
    print(f"Output neurons: {lm.params.num_output_neurons}")

    # Process some text
    text_to_process = "hello"
    print(f"\nProcessing text: '{text_to_process}'")
    lm.process_text(text_to_process)

    # Predict the next character
    next_char = lm.predict_next_char(text_to_process)
    print(f"Predicted next character: '{next_char}'")

    # You can continue processing and predicting
    another_text = " world"
    print(f"\nProcessing more text: '{another_text}'")
    lm.process_text(another_text)

    next_char_after_more = lm.predict_next_char(another_text)
    print(f"Next character after more processing: '{next_char_after_more}'")
