import os
import torch
from datasets import load_from_disk
from transformers import LayoutLMv2ForSequenceClassification
from transformers import TrainingArguments, Trainer
from src.DocumindAI.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.config = config

    def load_encoded_dataset(self):
        print("Loading encoded dataset from disk")
        base_path = self.config.data_path

        self.encoded_dataset = {
            'train': load_from_disk(os.path.join(base_path, "train")),
            'val': load_from_disk(os.path.join(base_path, "val")),
            'test': load_from_disk(os.path.join(base_path, "test"))
        }

        for split_name in self.encoded_dataset:
            self.encoded_dataset[split_name].set_format(type='torch')

        print("✅ Encoded dataset successfully loaded and formatted!")

    def initialize_model(self):
        print("Initializing LayoutLMv2 model")
        model_name = self.config.model
        num_labels = self.config.num_labels

        self.model = LayoutLMv2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        print("✅ Model initialized successfully!")

    def setup_trainer(self):
        print("Creating Trainer instance")
        self.training_args = TrainingArguments(
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            learning_rate=self.config.learning_rate,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            remove_unused_columns=self.config.remove_unused_columns,
            report_to=None
        )
        self.trainer = Trainer(
            model=self.config.model,
            args=self.training_args,
            train_dataset=self.encoded_dataset["train"],
            eval_dataset=self.encoded_dataset["val"],
        )
        print("Trainer ready!")

    def train_model(self):
        print("Starting model training...")
        self.trainer.train()
        print("Training completed!")

    def save_model(self):
        save_path = self.config.root_dir
        os.makedirs(save_path, exist_ok=True)

        print(f"Saving model to: {save_path}")
        self.model.save_pretrained(save_path)
        self.trainer.save_model(save_path)

        print("Model and trainer saved successfully!")

    def train(self):
        print("Running full model training pipeline")
        self.load_encoded_dataset()
        self.initialize_model()
        self.setup_trainer()
        self.train_model()
        self.save_model()
        print(" Model training pipeline completed successfully!")
