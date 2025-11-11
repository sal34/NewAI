from neuraxon import LanguageModel, load_network
from tokenizer import Tokenizer
import argparse

def generate_text(model: LanguageModel, tokenizer: Tokenizer, seed_text: str, length: int = 100) -> str:
    """
    Generates text using the trained Neuraxon language model.

    Args:
        model: The trained language model.
        tokenizer: The tokenizer.
        seed_text: The initial text to start generation.
        length: The number of characters to generate.

    Returns:
        The generated text.
    """
    print(f"Generating text with seed: '{seed_text}'")
    generated_text = seed_text

    # Use the seed text to set the initial state of the network
    model.process_text(seed_text)

    current_char = seed_text[-1]
    for _ in range(length):
        # Predict the next character based on the current state
        next_char = model.predict_next_char(current_char)

        if not next_char: # Stop if the model predicts an unknown character
            break

        generated_text += next_char
        current_char = next_char

        # Process the newly generated character to update the network state for the next prediction
        model.process_text(next_char)

    return generated_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate text with a trained Neuraxon language model.")
    parser.add_argument("--model_path", type=str, default="neuraxon_lm.json", help="Path to the trained model file.")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.json", help="Path to the tokenizer file.")
    parser.add_argument("--seed_text", type=str, default="The", help="The seed text to start generation.")
    parser.add_argument("--length", type=int, default=200, help="The number of characters to generate.")

    args = parser.parse_args()

    # Load the tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = Tokenizer.load(args.tokenizer_path)
    except FileNotFoundError:
        print(f"Error: Tokenizer file not found at {args.tokenizer_path}")
        print("Please run train.py first to create a tokenizer.")
        exit(1)

    # Load the trained model
    print("Loading trained model...")
    try:
        model = LanguageModel.load(args.model_path, tokenizer)
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        print("Please run train.py first to train and save a model.")
        exit(1)


    # Generate text
    generated_output = generate_text(model, tokenizer, args.seed_text, args.length)

    print("\n--- Generated Text ---")
    print(generated_output)
    print("----------------------")
