from transformers import AutoModelForTokenClassification, AutoTokenizer
from peft import PeftModel
import torch

# Path to the saved LoRA model
model_path = "AmelieSchreiber/esm2_t33_650M_qlora_binding_16M"  #Qlora optimed model
# ESM2 base model
base_model_path = "facebook/esm2_t33_650M_UR50D"

# Load the model
base_model = AutoModelForTokenClassification.from_pretrained(base_model_path)
loaded_model = PeftModel.from_pretrained(base_model, model_path)

# Ensure the model is in evaluation mode
loaded_model.eval()

# Load the tokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Protein sequence for inference
protein_sequence = "MAVPETRPNHTIYINNLNEKIKKDELKKSLHAIFSRFGQILDILVSRSLKMRGQAFVIFKEVSSATNALRSMQGFPFYDKPMRIQYAKTDSDIIAKMKGT"  # Replace with your actual sequence

# Tokenize the sequence
inputs = loaded_tokenizer(protein_sequence, return_tensors="pt", truncation=True, max_length=1024, padding='max_length')

# Run the model
with torch.no_grad():
    logits = loaded_model(**inputs).logits

# Get predictions
tokens = loaded_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])  # Convert input ids back to tokens
predictions = torch.argmax(logits, dim=2)

# Define labels
id2label = {
    0: "No binding site",
    1: "Binding site"
}

# Print the predicted labels for each token
for token, prediction in zip(tokens, predictions[0].numpy()):
    if token not in ['<pad>', '<cls>', '<eos>']:
        print((token, id2label[prediction]))
