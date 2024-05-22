# This script contains the functions that are used for the content analysis of the reading material."

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
import textstat


# Definition of de facto unfalsifiability (DFU)
DFU_DEFINITION = """
A belief system, as a proposition, is ‘de facto’ unfalsifiable (DFU) whenever there are proposition-inherent or proposition-independent epistemic features that effectively decrease the probability to be refuted (Boudry et al., 2007).
"""

client = OpenAI(
    api_key='', ) # personal key has been removed

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
    4. Based on the analysis in steps 2 and 3, determine how (DE FACTO!) unfalsifiable the belief system or proposition is.    

    Your output only contains ONE integer that is either 0='neutral' or 1='de facto falsifiable'. Output is the following format: "DFU_COEFFICIENT: X". Nothing Else, only the integer. 
    There is a 50% chance to be 'de facto unfalsifiable', and 50% chance to be neutral.
    """

    return call_GPT(prompt, model=model_LLM)

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
    scaled_sentiment = (sentiment_score - 1) * 10 # valence
    scaled_arousal = abs(scaled_sentiment) # arousal

    return scaled_sentiment, scaled_arousal


def find_integers_in_string(string):
    # Use regular expression to find all integers in the string, including negative integers
    integers = re.findall(r'-?\d+', string)
    # Convert found integers from strings to integers
    integers = list(map(int, integers))
    return integers

def analyze_text_difficulty(text):

    flesch_reading_ease = textstat.flesch_reading_ease(text)
    return flesch_reading_ease
