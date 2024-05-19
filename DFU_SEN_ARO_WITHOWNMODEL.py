"This script has the same functionality as 'Content_Analysis_DFU.py', but now uses a fine-tuned RO-BERT-a to extract the DFU-coefficient."

from textblob import TextBlob
from sklearn.metrics import mean_squared_error
import numpy as np
import re
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.preprocessing import MinMaxScaler
import SUPERVISED_DATASET

# Load the trained model and tokenizer
model_path = "trained_model"
dfu_model = AutoModelForSequenceClassification.from_pretrained(model_path)
dfu_tokenizer = AutoTokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dfu_model.to(device)

# Function to preprocess sentences and make predictions
def predict_dfu_coefficient(sentences):
    encodings = dfu_tokenizer(sentences, truncation=True, padding=True, return_tensors='pt').to(device)
    dfu_model.eval()
    with torch.no_grad():
        outputs = dfu_model(**encodings)
        predictions = outputs.logits.squeeze().cpu().numpy()
    return predictions

# Function to scale the DFU_COEFFICIENT values from 0-1 to 1-9
def scale_dfu_coefficient(predictions):
    scaler = MinMaxScaler(feature_range=(1, 9))
    scaled_predictions = scaler.fit_transform(predictions.reshape(-1, 1)).flatten()
    return scaled_predictions

# Example function to use the model for predicting DFU coefficients of sentences
def predict_dfu_for_sentences(sentences):
    dfu_predictions = predict_dfu_coefficient(sentences)
    scaled_dfu_predictions = scale_dfu_coefficient(dfu_predictions)
    return scaled_dfu_predictions

# Load the sentiment analysis model and tokenizer
sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

def analyze_sentiments(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = sentiment_model(**inputs)
    scores = outputs.logits.detach().cpu().numpy()
    sentiment_score = np.argmax(scores)  # The model scores are 0 to 2 for negative, neutral, positive

    # Map the sentiment score to a scale of -10 to 10
    # sentiment_score of 0 maps to -10, 1 maps to 0, 2 maps to 10
    scaled_sentiment = (sentiment_score - 1) * 10
    scaled_arousal = abs(scaled_sentiment)

    return scaled_sentiment, scaled_arousal

# Function to find integers in a string (placeholder, replace with actual function if needed)
def find_integers_in_string(string):
    integers = re.findall(r'-?\d+', string)
    integers = list(map(int, integers))
    return integers

# Function to run extractions sequentially
def run_extractions_sequentially(sentences):
    list_metrics = []
    DFU_average_scores = []
    SENTIMENT_scores = []

    dfu_coefficients = predict_dfu_for_sentences(sentences)

    for i,sentence in enumerate(sentences):
        dfu_coefficient = dfu_coefficients[i]
        sentiment, arousal = analyze_sentiments(sentence)

        list_metrics.append([dfu_coefficient, sentiment, arousal, sentence])

        DFU_average_scores.append(dfu_coefficient)
        SENTIMENT_scores.append(sentiment)

    # Calculate averages for DFU list and sentiment list
    DFU_average = np.mean(DFU_average_scores)
    SENTIMENT_average = np.mean(SENTIMENT_scores)

    return list_metrics, DFU_average, SENTIMENT_average

def print_3_metrics(dfu_coefficient, sentiment, arousal, sentence, text_prediction, sentence_type):
    print(f"Sentence type: {sentence_type}")
    print(f"Sentence: {sentence}\nDFU_COEFFICIENT: {dfu_coefficient} (ranges from 1 to 9)\nSENTIMENT_COEFFICIENT: {sentiment} (ranges from -10 to 10)\nAROUSAL_COEFFICIENT: {arousal} (ranges from 0 to 10)\n{text_prediction}\n\n")

def test_sentences(list_metrics, text_sentence_type, text_prediction, sentence_type):
    print(text_sentence_type)
    for dfu, pol, aro, sentence in list_metrics:
        print_3_metrics(dfu, pol, aro, sentence, text_prediction, sentence_type)

# Load example sentences
negative_sentences, neutral_sentences, positive_sentences, falsifiable_sentences, unfalsifiable_sentences = SUPERVISED_DATASET.get_example_sentences()

# Expectations for DFU_score (ranges from 1 to 9)
expectation_FAL_sentences = 9
expectation_UNF_sentences = 1

# Expectations for sentiment_score (ranges from -10 to 10)
expectation_NEG_sentences = -10
expectation_NEU_sentences = 0
expectation_POS_sentences = 10

# Test the sentences
list_metrics_FAL, DFU_average_FAL, _ = run_extractions_sequentially(falsifiable_sentences)
text_prediction = f"Predicted DFU_coefficient: {expectation_FAL_sentences}"
test_sentences(list_metrics_FAL, "TESTING THE CLEARLY FALSIFIABLE SENTENCES:", text_prediction, sentence_type='falsifiable')

list_metrics_UNFAL, DFU_average_UNF, _ = run_extractions_sequentially(unfalsifiable_sentences)
text_prediction = f"Predicted DFU_coefficient: {expectation_UNF_sentences}"
test_sentences(list_metrics_UNFAL, "TESTING THE CLEARLY UNFALSIFIABLE SENTENCES:", text_prediction, sentence_type='unfalsifiable')

list_metrics_NEG, _, SENTIMENT_average_NEG = run_extractions_sequentially(negative_sentences)
text_prediction = f"Predicted SENTIMENT_coefficient: {expectation_NEG_sentences}"
test_sentences(list_metrics_NEG, "TESTING THE CLEARLY NEGATIVE SENTENCES:", text_prediction, sentence_type='negative')

list_metrics_POS, _, SENTIMENT_average_POS = run_extractions_sequentially(positive_sentences)
text_prediction = f"Predicted SENTIMENT_coefficient: {expectation_POS_sentences}"
test_sentences(list_metrics_POS, "TESTING THE CLEARLY POSITIVE SENTENCES:", text_prediction, sentence_type='positive')

list_metrics_NEU, _, SENTIMENT_average_NEU = run_extractions_sequentially(neutral_sentences)
text_prediction = f"Predicted SENTIMENT_coefficient: {expectation_NEU_sentences}"
test_sentences(list_metrics_NEU, "TESTING THE NEUTRAL SENTENCES:", text_prediction, sentence_type='neutral')

# COMPARE MODELS WITH THIS CODE; the smaller the RMSE the better:
def calculate_rmse(predicted, expected):
    return np.sqrt(mean_squared_error([predicted], [expected]))

# Calculate RMSE for DFU scores
RMSE_fal = round(calculate_rmse(DFU_average_FAL, expectation_FAL_sentences), 4)
RMSE_unfal = round(calculate_rmse(DFU_average_UNF, expectation_UNF_sentences), 4)

# Calculate RMSE for sentiment scores
RMSE_neg = round(calculate_rmse(SENTIMENT_average_NEG, expectation_NEG_sentences), 4)
RMSE_neu = round(calculate_rmse(SENTIMENT_average_NEU, expectation_NEU_sentences), 4)
RMSE_pos = round(calculate_rmse(SENTIMENT_average_POS, expectation_POS_sentences), 4)

# Create a bar plot for RMSE values
Modelfor_DFU_extraction = 'finetuned-roberta'
Modelfor_SENTIMENT_extraction = 'twitter-roberta'

# Print performance comparison texts
performance_comparison_DFU_text = (
    f"Performance of the language model '{Modelfor_DFU_extraction}' to distinguish between falsifiable and unfalsifiable texts: \n"
    f"DFU_coefficient ranges from 1 to 9 (maximum difference is 8):"
    f"RMSE for clearly falsifiable sentences: {RMSE_fal}\n"
    f"RMSE for clearly unfalsifiable sentences: {RMSE_unfal}\n"
)

performance_comparison_SENTIMENT_text = (
    f"Performance of the language model {Modelfor_SENTIMENT_extraction} to distinguish between negative and positive valences: \n"
    f"SENTIMENT_coefficient ranges from -10 to 10 (maximum difference is 20):\n"
    f"RMSE for clearly negative sentences: {RMSE_neg}\n"
    f"RMSE for neutral sentences: {RMSE_neu}\n"
    f"RMSE for positive sentences: {RMSE_pos}\n"
)

print(performance_comparison_DFU_text)
print(performance_comparison_SENTIMENT_text)

# Normalizing the RMSE values to percentages for consistent interpretation
def normalize_rmse(rmse, scale_max):
    return (rmse / scale_max) * 100

# Normalize RMSE values based on their respective scales
RMSE_fal_normalized = normalize_rmse(RMSE_fal, 9)  # DFU scale is from 1 to 9
RMSE_unfal_normalized = normalize_rmse(RMSE_unfal, 9)  # DFU scale is from 1 to 9
RMSE_neg_normalized = normalize_rmse(RMSE_neg, 20)  # Sentiment scale is from -10 to 10
RMSE_neu_normalized = normalize_rmse(RMSE_neu, 20)  # Sentiment scale is from -10 to 10
RMSE_pos_normalized = normalize_rmse(RMSE_pos, 20)  # Sentiment scale is from -10 to 10

labels = [
    f'Falsifiable\n({Modelfor_DFU_extraction})',
    f'Unfalsifiable\n({Modelfor_DFU_extraction})',
    f'Negative\n({Modelfor_SENTIMENT_extraction})',
    f'Neutral\n({Modelfor_SENTIMENT_extraction})',
    f'Positive\n({Modelfor_SENTIMENT_extraction})'
]
RMSE_values = [RMSE_fal_normalized, RMSE_unfal_normalized, RMSE_neg_normalized, RMSE_neu_normalized, RMSE_pos_normalized]

plt.figure(figsize=(12, 8))
bars = plt.bar(labels, RMSE_values, color=['blue', 'orange', 'red', 'green', 'purple'])

# Annotate each bar with the RMSE value
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}%',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom')

plt.figure(figsize=(10, 6))
plt.bar(labels, RMSE_values, color=['blue', 'orange', 'red', 'green', 'purple'])
plt.xlabel('Sentence Type')
plt.ylabel('RMSE (%)')
plt.title('RMSE Comparison for Different Sentence Types', weight='bold')  # interpretatie: de RMSE(%) is de gemiddelde discrepantie tussen de verwachte score en de geschatte score
plt.savefig(f'RMSE-comparison_{Modelfor_DFU_extraction}&{Modelfor_SENTIMENT_extraction}.png')  # Overwrites if already exists
plt.show()
