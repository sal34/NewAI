import torch
from datasets import load_dataset
from tqdm import tqdm
from neuraxon import LanguageModel, NetworkParameters, save_network
from tokenizer import Tokenizer

def train(model: LanguageModel, dataset, epochs: int = 1, save_path: str = "neuraxon_lm.json"):
    """
    Trains the Neuraxon language model.

    Args:
        model: The language model to train.
        dataset: The dataset to train on.
        epochs: The number of epochs to train for.
        save_path: The path to save the trained model.
    """
    print("Starting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        progress_bar = tqdm(dataset, desc=f"Epoch {epoch + 1}")
        for item in progress_bar:
            text = item['text']

            # The training is unsupervised in the sense that we just process the text
            # and let the network's internal plasticity rules do the learning.
            model.process_text(text)

            # Optional: Display some progress
            # progress_bar.set_postfix({"current_text_snippet": text[:20]})

    print("\nTraining complete.")
    # Save the trained model
    save_network(model, save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    # Load the dataset
    print("Loading dataset...")
    # Using streaming to avoid downloading the full dataset
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", 'cosmopedia-v2', split="train", streaming=True)

    # We need to create a corpus to initialize the tokenizer.
    # We'll take a small sample from the dataset for this.
    print("Building tokenizer from a sample of the dataset...")
    corpus_sample = [item['text'] for item in dataset.take(10)] # Using 10 samples to build a decent vocab
    tokenizer = Tokenizer(corpus=corpus_sample)
    tokenizer.save("tokenizer.json")
    print(f"Tokenizer created with vocabulary size: {tokenizer.vocab_size}")

    # Initialize the model
    print("Initializing language model...")
    # Let's define some parameters for a slightly larger network
    params = NetworkParameters(
        num_hidden_neurons=256,
        connection_probability=0.08,
        learning_rate=0.015
    )
    model = LanguageModel(tokenizer=tokenizer, params=params)

    # Train the model
    # We will only train on a small subset of the data for this example
    training_data = dataset.take(10) # Using 10 samples for training
    train(model, training_data, epochs=1)

    print("\nTo generate text with the trained model, run generate.py")
