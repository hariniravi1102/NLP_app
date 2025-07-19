# train_reward_model.py
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from sklearn.preprocessing import MinMaxScaler
import os

class FeedbackDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                input_text = f"Prompt: {obj['prompt']}\nOutput: {obj['output']}"
                label = sum([obj[k] for k in ["fluency", "coherence", "factuality", "relevance"]]) / 20.0
                self.samples.append((input_text, label))
        self.scaler = MinMaxScaler()
        labels = [x[1] for x in self.samples]
        self.scaler.fit([[x] for x in labels])
        self.samples = [(s[0], float(self.scaler.transform([[s[1]]])[0][0])) for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.float)
        }

class RewardModel(torch.nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.base = AutoModel.from_pretrained(base_model_name)
        self.value_head = torch.nn.Linear(self.base.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        rewards = self.value_head(pooled).squeeze()
        loss = None
        if labels is not None:
            loss = torch.nn.functional.mse_loss(rewards, labels)
        return {"loss": loss, "logits": rewards, "labels": labels}

if __name__ == "__main__":
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = RewardModel(model_name)

    dataset = FeedbackDataset("feedback_data.jsonl", tokenizer)
    args = TrainingArguments(
        output_dir="reward_model",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_steps=100,
        logging_steps=10,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    model_path = "reward_model/checkpoint-final"
    os.makedirs(model_path, exist_ok=True)
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
