import pickle
import json

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from bilstm_nn import BiLSTMClassifier
from preprocessing import tokenize_function

MODEL_OPTIONS = [
    "Naive Bayes",
    "SVM",
    "BiLSTM",
    "MiniLM L12",
    "RoBERTa",
]

ID2LABEL = {0: "Hate", 1: "Offensive", 2: "Neither"}

CLASS_NAME = list(ID2LABEL.values())

# Traditional ML Models


class MLModels:
    def __init__(self, model_str):
        if model_str == "nb":
            with open("./results/naive_bayes/model.pkl", "rb") as f:
                self.model = pickle.load(f)
        elif model_str == "svm":
            with open("./results/svm/model.pkl", "rb") as f:
                self.model = pickle.load(f)
        else:
            raise ValueError(f"no model {model_str}")

        with open("./results/naive_bayes/tfidf.pkl", "rb") as f:
            self.tfidf = pickle.load(f)

    def predict(self, text):
        vec = self.tfidf.transform([text])

        proba = self.model.predict_proba(vec)[0]

        scores = {
            ID2LABEL[int(class_id)]: round(float(prob), 4)
            for class_id, prob in zip(self.model.classes_, proba)
        }

        pred_label = max(scores, key=scores.get)

        return {
            "label": pred_label,
            "score": scores[pred_label],
            "scores": scores,
        }


# BiLSTM Model


class BiLSTM:
    UNK_IDX = 1
    MAX_LEN = 256

    def __init__(self):
        with open("./results/bilstm/vocab.json") as f:
            self.vocab = json.load(f)

        with open("./results/bilstm/id2label.json") as f:
            self.id2label = json.load(f)

        with open("./results/bilstm/config.json") as f:
            self.cfg = json.load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = BiLSTMClassifier(
            vocab_size=self.cfg["vocab_size"],
            embed_dim=self.cfg["embed_dim"],
            hidden_dim=self.cfg["hidden_dim"],
            num_layers=self.cfg["num_layers"],
            num_classes=self.cfg["num_classes"],
            dropout=self.cfg["dropout"],
        ).to(self.device)

        model_dict = torch.load(
            "./results/bilstm/best_model.pt", map_location=self.device
        )

        self.model.load_state_dict(model_dict)
        self.model.eval()

    def predict(self, text):
        tokens = tokenize_function(text)[: self.MAX_LEN]

        ids = torch.tensor(
            [[self.vocab.get(t, self.UNK_IDX) for t in tokens]], dtype=torch.long
        ).to(self.device)

        length = torch.tensor([len(tokens)])

        with torch.no_grad():
            logits = self.model(ids, length)
            probs = torch.softmax(logits, dim=1)[0]

        scores = {
            ID2LABEL[i]: round(float(probs[i].item()), 4) for i in range(len(ID2LABEL))
        }

        pred_label = max(scores, key=scores.get)

        return {
            "label": pred_label,
            "score": scores[pred_label],
            "scores": scores,
        }


# Transformer Model


class TransformerModel:
    def __init__(self, model_str):
        if model_str == "minilm":
            output_path = "./results/mini_lm"
        elif model_str == "roberta":
            output_path = "./results/roberta"
        else:
            raise ValueError(f"no model {model_str}")

        tokenizer = AutoTokenizer.from_pretrained(output_path)
        model = AutoModelForSequenceClassification.from_pretrained(output_path)

        model.config.id2label = {
            0: "Hate",
            1: "Offensive",
            2: "Neither",
        }
        model.config.label2id = {v: k for k, v in model.config.id2label.items()}

        self.classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None,
        )

    def predict(self, text):
        result = self.classifier(text)[0]

        scores = {item["label"]: round(float(item["score"]), 4) for item in result}

        pred_label = max(scores, key=scores.get)

        return {
            "label": pred_label,
            "score": scores[pred_label],
            "scores": scores,
        }


# Load to memory

MODELS = {
    "Naive Bayes": MLModels(model_str="nb"),
    "SVM": MLModels(model_str="svm"),
    "BiLSTM": BiLSTM(),
    "MiniLM L12": TransformerModel(model_str="minilm"),
    "RoBERTa": TransformerModel(model_str="roberta"),
}

# expected output:
#
# {
#   "label": "Hate",
#   "score": 0.8208,
#   "scores": {
#      "Hate": 0.8208,
#      "Offensive": 0.1201,
#      "Neither": 0.0591
#    }
# }


def predict_text(text, model_name):
    selected_model = MODELS[model_name]
    return selected_model.predict(text)
