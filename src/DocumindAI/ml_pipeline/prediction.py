import torch
from transformers import LayoutLMv3ForSequenceClassification, AutoProcessor
from PIL import Image
import os
import json
from src.DocumindAI.config.configuration import ConfigurationManager



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename
        self.config = ConfigurationManager().get_evaluation_config()


    
    def predict(self):
        processor = AutoProcessor.from_pretrained(self.config.model_path)
        model = LayoutLMv3ForSequenceClassification.from_pretrained(self.config.model_path)
        model.eval()

        image = Image.open(self.filename).convert("RGB")

        encoding = processor(
            images=image,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        encoding = {k:v for k,v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)

        logits = outputs.logits
        probs = torch.softmax(logits,dim=-1)

        predicted_id = int(probs.argmax(dim=-1).item())
        confidence = float(probs.max().item())

        with open(os.path.join("config","id2label.json"),"r") as f:
            id2label = json.load(f)
        
        id2label = {int(k):v for k,v in id2label.items()}

        predicted_label = id2label[predicted_id]  

        return predicted_label, confidence  



