from transformers import BertTokenizer, BertForSequenceClassification
import torch
import joblib

# Load the model, tokenizer, and category mapping
model_path = "./categorization_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
category_mapping = joblib.load("category_mapping.pkl")

# Reverse the category mapping for decoding predictions
reverse_category_mapping = {v: k for k, v in category_mapping.items()}


# Function to predict the category of a ticket
def predict_category(text):
    # Tokenize the input text
    inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    # Decode the prediction to the original category
    category = reverse_category_mapping.get(prediction, "Unknown")
    return category

# Example usage
test_tickets = [
    "China eCommerce website giving error during checkout.",
    "PKB inventory not updated",
    "Reservation did not updated in global Trade",
    "I am unable to open the products featured in the HP carousel for the German site.",
 ]

# Predict and print the results
for ticket in test_tickets:
    category = predict_category(ticket)
    print(f"Ticket: {ticket}\nPredicted Category: {category}\n")



