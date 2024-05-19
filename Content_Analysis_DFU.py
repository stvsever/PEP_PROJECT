"This script analyzes the de facto unfalsifiability (DFU) and sentiment of various sentences using openai's generative pre-trained transformers for DFU quantification AND a sentiment analysis model (Roberta,textblob,gpt-3.5-turbo-0125,gpt-4) for sentiment evaluation."
"It runs these analyses in parallel, calculates average DFU and sentiment scores, and compares the predicted scores against expected values to compute RMSE (Root Mean Squared Error) for different sentence types."
"Finally, it visualizes the RMSE values in a bar plot to assess the models' performance."

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from textblob import TextBlob
from sklearn.metrics import mean_squared_error
import numpy as np
import re
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import SUPERVISED_DATASET

# load sentences (five lists of different sentence types)
negative_sentences, neutral_sentences,positive_sentences,falsifiable_sentences,unfalsifiable_sentences = SUPERVISED_DATASET.get_example_sentences()

# Definition of de facto unfalsifiability (DFU)
DFU_DEFINITION = """
A belief system, as a proposition, is ‘de facto’ unfalsifiable (DFU) whenever there are proposition-inherent or proposition-independent epistemic features that effectively decrease the probability to be refuted (Boudry et al., 2007).
"""

client = OpenAI(
    api_key='', )

model_LLM = "gpt-4o"

def call_GPT(GPT_prompt, model=model_LLM):
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": GPT_prompt}],
        stream=True,
    )

    result_text = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            result_text += chunk.choices[0].delta.content

    return result_text.strip()


def extract_DFU_coefficient(text):
    prompt = f"""
    Given the definition: {DFU_DEFINITION}

    Analyze the following text and determine the 'de facto unfalsifiability' coefficient (DFU_coefficient). 

    Text: "{text}"

    Stepwise procedure for critical content analysis:
    1. Identify the main belief system or proposition in the text.
    2. Examine any proposition-inherent epistemic features that may decrease the probability of the proposition being refuted.
    3. Examine any proposition-independent epistemic features that may decrease the probability of the proposition being refuted.
    4. Based on the analysis in steps 2 and 3, determine how unfalsifiable the belief system or proposition is.    

    Your output only contains ONE integer that ranges from 1 to 9. It is a binary semantic (Likert) scale where 1='extremely DFunfalsifiable' and 9='extremely DFfalsifiable'. Output is the following format: "DFU_COEFFICIENT: X". Nothing Else, only the integer.
    """

    return call_GPT(prompt, model=model_LLM)


#def analyze_sentiments(text): # for textblob ; other analyze_sentiments(text) is for GPT's
    #analysis = TextBlob(text)
    #scaled_sentiment = analysis.sentiment.polarity * 10
    #scaled_arousal = abs(scaled_sentiment)
    #return scaled_sentiment, scaled_arousal

#def analyze_sentiments(text):
    #prompt = f"""
        #Analyze the following text and determine the valence coefficient.

        #Text: "{text}"

        #Your output only contains ONE integer that ranges from -10 to 10. It is a binary semantic (Likert) scale where -10='extremely negative' and 10='extremely positive'. Output is the following format: "VALENCE_COEFFICIENT: X". Nothing else, only the integer as output.
        #"""


    #sentiment = find_integers_in_string(call_GPT(prompt, model=model_LLM))[0]  # Assuming there's only one integer
    #arousal = abs(sentiment)

    #return sentiment,arousal

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

def analyze_sentiments(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = outputs.logits.detach().numpy()
    sentiment_score = np.argmax(scores)  # The model scores are 0 to 2 for negative, neutral, positive

    # Map the sentiment score to a scale of -10 to 10
    # sentiment_score of 0 maps to -10, 1 maps to 0, 2 maps to 10
    scaled_sentiment = (sentiment_score - 1) * 10
    scaled_arousal = abs(scaled_sentiment)

    return scaled_sentiment, scaled_arousal


def find_integers_in_string(string):
    # Use regular expression to find all integers in the string, including negative integers
    integers = re.findall(r'-?\d+', string)
    # Convert found integers from strings to integers
    integers = list(map(int, integers))
    return integers


# Function to run the extraction in parallel
def run_extractions_in_parallel(sentences):
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_sentence = {executor.submit(extract_DFU_coefficient, sentence): sentence for sentence in sentences}

        list_metrics = []
        DFU_average_scores = []
        SENTIMENT_scores = []

        for future in as_completed(future_to_sentence):
            sentence = future_to_sentence[future]

            sentiment, arousal = analyze_sentiments(sentence)
            dfu_coefficient_output = future.result()
            dfu_coefficient = find_integers_in_string(dfu_coefficient_output)[0]  # Assuming there's only one integer

            list_metrics.append([dfu_coefficient, sentiment, arousal, sentence])
            DFU_average_scores.append(dfu_coefficient)
            SENTIMENT_scores.append(sentiment)

        # Calculate averages for DFU list and sentiment list
        DFU_average = np.mean(DFU_average_scores)
        SENTIMENT_average = np.mean(SENTIMENT_scores)

        return list_metrics, DFU_average, SENTIMENT_average

def print_3_metrics(dfu_coefficient, sentiment, arousal, sentence,text_prediction, sentence_type):
    print(f"Sentence type: {sentence_type}")
    print(f"Sentence: {sentence}\nDFU_COEFFICIENT: {dfu_coefficient} (ranges from 1 to 9)\nSENTIMENT_COEFFICIENT: {sentiment} (ranges from -10 to 10)\nAROUSAL_COEFFICIENT: {arousal} (ranges from 0 to 10)\n{text_prediction}\n\n")

def test_sentences(list_metrics, text_sentence_type,text_prediction,sentence_type):
    print(text_sentence_type)
    for dfu, pol, aro, sentence in list_metrics:
        print_3_metrics(dfu, pol, aro, sentence,text_prediction,sentence_type)

# Expectation values to calculate RMSE ; NOTE that, although extreme sentence types were manually made, they do not all truly match the extremity of their sentence type
# Expectations for DFU_score (ranges from 1 to 9)
expectation_FAL_sentences = 9
expectation_UNF_sentences = 1

# Expectations for sentiment_score (ranges from -10 to 10)
expectation_NEG_sentences = -10
expectation_NEU_sentences = 0
expectation_POS_sentences = 10

# Test the sentences
list_metrics_FAL, DFU_average_FAL, _ = run_extractions_in_parallel(falsifiable_sentences)
text_prediction = f"Predicted DFU_coefficient: {expectation_FAL_sentences}"
test_sentences(list_metrics_FAL,"TESTING THE CLEARLY FALSIFIABLE SENTENCES:",text_prediction,sentence_type='falsifiable')

list_metrics_UNFAL, DFU_average_UNF, _ = run_extractions_in_parallel(unfalsifiable_sentences)
text_prediction = f"Predicted DFU_coefficient: {expectation_UNF_sentences}"
test_sentences(list_metrics_UNFAL,"TESTING THE CLEARLY UNFALSIFIABLE SENTENCES:",text_prediction,sentence_type='unfalsifiable')

list_metrics_NEG, _, SENTIMENT_average_NEG = run_extractions_in_parallel(negative_sentences)
text_prediction = f"Predicted SENTIMENT_coefficient: {expectation_NEG_sentences}"
test_sentences(list_metrics_NEG,"TESTING THE CLEARLY NEGATIVE SENTENCES:",text_prediction,sentence_type='negative')

list_metrics_POS, _, SENTIMENT_average_POS = run_extractions_in_parallel(positive_sentences)
text_prediction = f"Predicted SENTIMENT_coefficient: {expectation_POS_sentences}"
test_sentences(list_metrics_POS,"TESTING THE CLEARLY POSITIVE SENTENCES:",text_prediction,sentence_type='positive')

list_metrics_NEU, _, SENTIMENT_average_NEU = run_extractions_in_parallel(neutral_sentences)
text_prediction = f"Predicted SENTIMENT_coefficient: {expectation_NEU_sentences}"
test_sentences(list_metrics_NEU,"TESTING THE NEUTRAL SENTENCES:",text_prediction,sentence_type='neutral')

# COMPARE MODELS WITH THIS CODE ; the smaller the RMSE the better:
def calculate_rmse(predicted, expected):
    return np.sqrt(mean_squared_error([predicted], [expected]))

# COMPARE MODELS WITH THIS CODE ; the smaller the RMSE (root mean squared error) the better:

# Calculate RMSE for DFU scores
RMSE_fal = round(calculate_rmse(DFU_average_FAL, expectation_FAL_sentences),4)
RMSE_unfal = round(calculate_rmse(DFU_average_UNF, expectation_UNF_sentences),4)

# Calculate RMSE for sentiment scores
RMSE_neg = round(calculate_rmse(SENTIMENT_average_NEG, expectation_NEG_sentences),4)
RMSE_neu = round(calculate_rmse(SENTIMENT_average_NEU, expectation_NEU_sentences),4)
RMSE_pos = round(calculate_rmse(SENTIMENT_average_POS, expectation_POS_sentences),4)

# Create a bar plot for RMSE values
Modelfor_DFU_extraction = model_LLM
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
RMSE_fal_normalized = normalize_rmse(RMSE_fal, 9)    # DFU scale is from 1 to 9
RMSE_unfal_normalized = normalize_rmse(RMSE_unfal, 9)  # DFU scale is from 1 to 9
RMSE_neg_normalized = normalize_rmse(RMSE_neg, 20)    # Sentiment scale is from -10 to 10
RMSE_neu_normalized = normalize_rmse(RMSE_neu, 20)    # Sentiment scale is from -10 to 10
RMSE_pos_normalized = normalize_rmse(RMSE_pos, 20)    # Sentiment scale is from -10 to 10

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
plt.title('RMSE Comparison for Different Sentence Types',weight='bold') # interpretatie: de RMSE(%) is de gemiddelde discrepantie tussen de verwachte score en de geschatte score
plt.savefig(f'RMSE-comparison_{Modelfor_DFU_extraction}&{Modelfor_SENTIMENT_extraction}.png') # Overwrites if already exists
plt.show()
