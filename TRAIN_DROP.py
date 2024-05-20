"In this script a 'for loop' loops through a list of drop-out rates to ultimately estimate model generalizibility."

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
import torch
import SUPERVISED_DATASET

# Load the datasets
old_data = SUPERVISED_DATASET.old_data()
new_data = SUPERVISED_DATASET.new_data()
extra_data = SUPERVISED_DATASET.extra_data()

# Combine the datasets into one DataFrame
combined_data = {
    "text": old_data["text"] + new_data["text"] + extra_data["text"],
    "dfu_coefficient": old_data["dfu_coefficient"] + new_data["dfu_coefficient"] + extra_data["dfu_coefficient"]
}

df = pd.DataFrame(combined_data)

# Display information about the DataFrame
print(f"Size: {df.size}")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print(f"Column names: {df.columns.tolist()}")
print(f"First few rows:\n{df.head()}")
print(f"Data types:\n{df.dtypes}")

# Function to compute Root Mean Squared Error
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Split off 30 sentences as the test set
df_train_valid, df_test = train_test_split(df, test_size=30, random_state=42)

# Drop rates
drop_rates = np.linspace(0, 0.95, 50)[::-1]

# Results
dataset_sizes = []
train_accuracies = []
valid_accuracies = []
test_accuracies = []

# Define tokenization and dataset creation functions
def tokenize_and_create_dataset(df):
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        df["text"], df["dfu_coefficient"], test_size=0.125, random_state=42
    )

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, return_tensors='pt')
    valid_encodings = tokenizer(valid_texts.tolist(), truncation=True, padding=True, return_tensors='pt')

    class DfuDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels.iloc[idx], dtype=torch.float)
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = DfuDataset(train_encodings, train_labels)
    valid_dataset = DfuDataset(valid_encodings, valid_labels)

    return train_dataset, valid_dataset, tokenizer

# Tokenize the test dataset
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
test_encodings = tokenizer(df_test["text"].tolist(), truncation=True, padding=True, return_tensors='pt')
test_labels = df_test["dfu_coefficient"]

class DfuTestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels.iloc[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

test_dataset = DfuTestDataset(test_encodings, test_labels)

# Training and evaluation loop
for drop_rate in drop_rates:
    # Create a temporary DataFrame with a subset of the data
    temp_df = df_train_valid.sample(frac=1 - drop_rate, random_state=42)

    # Tokenize and create datasets
    train_dataset, valid_dataset, tokenizer = tokenize_and_create_dataset(temp_df)

    # Model Definition using RoBERTa for regression
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=1)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.squeeze()
        rmse = root_mean_squared_error(labels, preds)
        return {
            "rmse": rmse,
        }

    # Adjusted Training Arguments for faster training
    training_args = TrainingArguments(
        output_dir="../results",
        num_train_epochs=8,  # Fixed number of epochs
        per_device_train_batch_size=20,  # Increased batch size
        per_device_eval_batch_size=20,  # Increased batch size
        warmup_steps=25,  # Reduced number of warmup steps
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    # Training
    trainer.train()

    # Evaluate the model on the training set
    train_outputs = trainer.predict(train_dataset)
    train_rmse = compute_metrics(train_outputs)["rmse"]
    train_accuracies.append(train_rmse)

    # Evaluate the model on the validation set
    eval_result = trainer.evaluate(eval_dataset=valid_dataset)
    valid_rmse = eval_result["eval_rmse"]
    valid_accuracies.append(valid_rmse)

    # Save dataset size and validation accuracy
    dataset_sizes.append(len(temp_df))

    # Move the encodings to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_encodings = {key: val.to(device) for key, val in test_encodings.items()}

    # Make predictions on the test dataset
    model.eval()
    with torch.no_grad():
        outputs = model(**test_encodings)
        predictions = outputs.logits.squeeze().cpu().numpy()

    # Compute test RMSE
    test_rmse = root_mean_squared_error(test_labels, predictions)
    test_accuracies.append(test_rmse)

    # Collect training and validation losses
    training_logs = trainer.state.log_history
    train_losses = [log["loss"] for log in training_logs if "loss" in log]
    eval_losses = [log["eval_loss"] for log in training_logs if "eval_loss" in log]

    # Ensure lengths of train_losses and eval_losses match epochs
    num_epochs = training_args.num_train_epochs
    train_losses = train_losses[:num_epochs]
    eval_losses = eval_losses[:num_epochs]
    epochs = range(1, num_epochs + 1)

    # Debugging prints to check lengths
    print(f"Drop rate: {drop_rate}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Training losses: {len(train_losses)}")
    print(f"Evaluation losses: {len(eval_losses)}")

    # Plot the training and validation RMSE over epochs
    if len(train_losses) == len(epochs) and len(eval_losses) == len(epochs):
        plt.figure(figsize=(14, 7))
        plt.plot(epochs, train_losses, label="Training RMSE", marker='o')
        plt.plot(epochs, eval_losses, label="Validation RMSE", marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.title(f"Training and Validation RMSE Over Epochs (Drop Rate: {drop_rate:.2f})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"training_and_validation_rmse_over_epochs_drop_rate_{drop_rate:.2f}.png")
        plt.show()
        plt.close()

    # Print detailed results for the last dropout rate value
    if drop_rate == drop_rates[-1]:
        print("Detailed results for the last dropout rate value:")
        for i, (text, true_label, pred) in enumerate(zip(df_test["text"].tolist(), test_labels.tolist(), predictions)):
            print(f"Text_{i+1}: {text}\nTrue DFU-coefficient: {true_label}\nPredicted DFU-coefficient: {pred:.2f}\n")

# Plot the results for dataset size vs validation RMSE
plt.figure(figsize=(10, 6))
plt.plot(dataset_sizes, test_accuracies, marker='o', linestyle='-', color='b')
plt.xlabel("Dataset Size")
plt.ylabel("RMSE")
plt.title("Model Accuracy vs Dataset Size")
plt.legend()
plt.grid(True)
plt.savefig("model_accuracy_vs_dataset_size_TES.png")
plt.show()
plt.close()

# Plot the results for drop rate vs validation RMSE
plt.figure(figsize=(10, 6))
plt.plot(drop_rates, test_accuracies, marker='o', linestyle='-', color='r')
plt.xlabel("Drop Rate")
plt.ylabel("RMSE")
plt.title("Model Accuracy vs Drop Rate")
plt.legend()
plt.grid(True)
plt.savefig("model_accuracy_vs_drop_rate_TES.png")
plt.show()
plt.close()
