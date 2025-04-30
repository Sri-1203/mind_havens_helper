import torch
from transformers import BertTokenizer, BertModel
from transformers import BertForSequenceClassification


model_name = "sri1208/mental_health_classifier"  # Change this to another BERT model if needed
sa_tokenizer = BertTokenizer.from_pretrained(model_name)
LABELS = ['Normal', 'Depression', 'Suicidal', 'Anxiety', 'Bipolar']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    # Recommended: Let transformers handle safetensors automatically
    sa_model = BertForSequenceClassification.from_pretrained(
        model_name,
        #local_files_only=True
    ).to(device)
    sa_model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    # Optional: Verify safetensors metadata
    # try:
    #     with safe_open(f"{MODEL_DIR}/model.safetensors", framework="pt") as f:
    #         print("Metadata keys:", list(f.keys()))
    #         print("File metadata:", f.metadata())
    # except Exception as safetensor_error:
    #     print(f"Safetensors validation failed: {safetensor_error}")
    exit()

def predict(text):
    try:
        inputs = sa_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            outputs = sa_model(**inputs)
            logits = outputs.logits

        # For multi-label classification
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        predictions = {LABELS[i]: float(probs[i] > 0.5) for i in range(len(LABELS))}
        return {
            "predictions": predictions,
            "probabilities": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
        }

    except Exception as e:
        return {"error": str(e)}
    
example_text = (
        "For the past 4 years it feels like life has thrown me constant pain and misery and I am not sure I can take it anymore, I am only a young lad 18 and I just feel like I cannot live for another 50 years with the way I am feeling right now. I am so frightened of being an adult comparing myself with everyone around me maintain friendships and relationships with my family. I feel so ungrateful saying all of this but its just the way I feel. I am so frightened of the pain of death but I would love to die if any of that makes sense, just needed to get this off of my chest. Why is nothing going my way at all in life"
    )

result = predict(example_text)
#print("Predictions:", result["predictions"])
#print("Probabilities:", result["probabilities"])