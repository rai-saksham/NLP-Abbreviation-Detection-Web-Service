import logging
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from fastapi_cache.backends.inmemory import InMemoryBackend
import datetime
import uvicorn
import os
import json
from fastapi_cache.decorator import cache
from fastapi_cache import FastAPICache

CONFIG = {}
MODEL = None

logger = logging.getLogger(__name__)

app = FastAPI()


def get_model():
    return MODEL


class Model:
    def __init__(self, model_name: str):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Define the class labels (Adjust this list based on your specific use case)
        self.classes = ['B-AC', 'B-LF', 'B-O', 'I-LF']
        self.id2label = {i: label for i, label in enumerate(self.classes)}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.model = self._load_model(model_name)
        self.model.to(self.device)

    def _load_model(self, model_name: str):
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, id2label=self.id2label, label2id=self.label2id)
        return model

    def predict(self, text: str):
        logging.info(f"Predicting for {text}")
        pipe = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="first",
                        device=0 if torch.cuda.is_available() else -1)
        result = pipe(text)
        logging.info(f"Predicting result: {result}")
        return result

    def format_predictions(self, predictions):
        formatted_results = []
        for res in predictions:
            formatted_results.append({
                "label": res["entity_group"],
                "word": res["word"],
                "score": float(res["score"])  # Convert numpy float to standard float
            })
        return formatted_results


class TextRequest(BaseModel):
    text: str


@app.post("/predict")
def predict_text(request: TextRequest, model: Model = Depends(get_model)):
    try:
        logging.info(f"Predicting for the input text: {request.text}")

        # Perform prediction
        results = model.predict(request.text)

        # Format the results
        formatted_results = model.format_predictions(results)

        return {"result": formatted_results}
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction")


@app.get("/predict_static")
def predict_static(model: Model = Depends(get_model)):
    input_sentence = "Abbreviations: GEMS, Global Enteric Multicenter Study; VIP, ventilated improved pit."
    try:
        logging.info(f"Predicting for the static sentence: {input_sentence}")

        # Perform prediction
        results = model.predict(input_sentence)

        # Format the results
        formatted_results = model.format_predictions(results)

        return {"result": formatted_results}
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction")


if __name__ == "__main__":
    # Define the root directory and construct the path to the config file
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(root_dir, "config", "config.json")

    with open(config_path) as json_file:
        CONFIG = json.load(json_file)

    log_file = CONFIG.get("log_file", os.path.join(root_dir, "logs", "service.log"))

    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s')

    start_time = datetime.datetime.now()
    model_path = os.path.join(CONFIG["MODEL_NAME"])
    MODEL = Model(model_path)

    FastAPICache.init(InMemoryBackend())
    logger.info("Service started successfully in %s", datetime.datetime.now() - start_time)
    uvicorn.run(app, host="0.0.0.0", port=CONFIG.get("port", 8000))

