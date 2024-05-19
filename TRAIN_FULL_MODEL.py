import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split, KFold
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

# Additional metrics for evaluation
def mean_absolute_error_metric(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def r2_score_metric(y_true, y_pred):
    return r2_score(y_true, y_pred)

# Split off 30 sentences as the test set
df_train_valid, df_test = train_test_split(df, test_size=30, random_state=42)

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

# Tokenize the training and validation datasets
train_dataset, valid_dataset, tokenizer = tokenize_and_create_dataset(df_train_valid)

# Tokenize the test dataset
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

# Model Definition using RoBERTa for regression
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=1)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.squeeze()
    rmse = root_mean_squared_error(labels, preds)
    mae = mean_absolute_error_metric(labels, preds)
    r2 = r2_score_metric(labels, preds)
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

# Adjusted Training Arguments for faster training
training_args = TrainingArguments(
    output_dir="../results",
    num_train_epochs=10,  # Fixed number of epochs
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
train_metrics = compute_metrics(train_outputs)

# Evaluate the model on the validation set
eval_result = trainer.evaluate(eval_dataset=valid_dataset)
valid_metrics = eval_result

# Move the encodings to the same device as the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
test_encodings = {key: val.to(device) for key, val in test_encodings.items()}

# Make predictions on the test dataset
model.eval()
with torch.no_grad():
    outputs = model(**test_encodings)
    predictions = outputs.logits.squeeze().cpu().numpy()

# Compute test metrics
test_rmse = root_mean_squared_error(test_labels, predictions)
test_mae = mean_absolute_error_metric(test_labels, predictions)
test_r2 = r2_score_metric(test_labels, predictions)

# Collect training and validation losses
training_logs = trainer.state.log_history
train_losses = [log["loss"] for log in training_logs if "loss" in log]
eval_losses = [log["eval_loss"] for log in training_logs if "eval_loss" in log]

# Ensure lengths of train_losses and eval_losses match epochs
num_epochs = training_args.num_train_epochs
train_losses = train_losses[:num_epochs]
eval_losses = eval_losses[:num_epochs]
epochs = range(1, num_epochs + 1)

# Plot the training and validation RMSE over epochs
plt.figure(figsize=(14, 7))
plt.plot(epochs, train_losses, label="Training Loss", marker='o')
plt.plot(epochs, eval_losses, label="Validation Loss", marker='o')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("training_and_validation_loss_over_epochs.png")
plt.show()
plt.close()

# Plot true labels vs predicted values for the test set
plt.figure(figsize=(10, 6))
plt.scatter(test_labels, predictions, color='blue', alpha=0.6)
plt.plot([min(test_labels), max(test_labels)], [min(test_labels), max(test_labels)], color='red', linestyle='--')
plt.xlabel("True DFU-coefficient")
plt.ylabel("Predicted DFU-coefficient")
plt.title("True vs Predicted DFU-coefficient")
plt.grid(True)
plt.savefig("true_vs_predicted_dfu_coefficient.png")
plt.show()
plt.close()

# Residual plot
residuals = test_labels - predictions
plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals, color='blue', alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted DFU-coefficient")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.savefig("residual_plot.png")
plt.show()
plt.close()

# Print detailed results
print("Detailed results for the test dataset:")
for i, (text, true_label, pred) in enumerate(zip(df_test["text"].tolist(), test_labels.tolist(), predictions)):
    print(f"Text_{i+1}: {text}\nTrue DFU-coefficient: {true_label}\nPredicted DFU-coefficient: {pred:.2f}\n")

print(f"Training RMSE: {train_metrics['rmse']}")
print(f"Validation RMSE: {valid_metrics['eval_rmse']}")
print(f"Test RMSE: {test_rmse}")

print(f"Training MAE: {train_metrics['mae']}")
print(f"Validation MAE: {valid_metrics['eval_mae']}")
print(f"Test MAE: {test_mae}")

print(f"Training R2: {train_metrics['r2']}")
print(f"Validation R2: {valid_metrics['eval_r2']}")
print(f"Test R2: {test_r2}")

# Save the trained model and tokenizer
trainer.save_model("./trained_model")
tokenizer.save_pretrained("./trained_model")
