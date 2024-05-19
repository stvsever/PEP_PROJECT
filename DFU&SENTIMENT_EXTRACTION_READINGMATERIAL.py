# This script is used to compute four metrics (1 measure of interest ; 3 confounding variables) of the reading material.
# Three models are used: "twitter-roberta-base-sentiment", "gpt-4o" and "Flesch's reading ease model".

import SUPERVISED_DATASET
import FUNCTIONS_FOR_ANALYSIS_READING_MATERIAL as RM_ANALYSIS
import numpy as np

reading_material = SUPERVISED_DATASET.paper_reading_material()

# Initialize lists
reading_material_DFU = []
reading_material_DIF = []
reading_material_SEN = []
reading_material_ARO = []


def calculate_4_metrics(sentences, output_file):
    with open(output_file, "w") as file:
        for sentence in sentences:
            # 1 DFU COEFFICIENT
            dfu_coefficient_output = RM_ANALYSIS.extract_DFU_coefficient(sentence)
            dfu_coefficient = RM_ANALYSIS.find_integers_in_string(dfu_coefficient_output)[0]

            # 2 READING DIFFICULTY COEFFICIENT
            reading_difficulty = RM_ANALYSIS.analyze_text_difficulty(sentence)

            # 3 & 4 VALENCE COEFFICIENT AND AROUSAL COEFFICIENT
            sentiment, arousal = RM_ANALYSIS.analyze_sentiments(sentence)

            # Print to console
            print(
                f"Sentence: {sentence}\nDFU Coefficient: {dfu_coefficient}\nReading Difficulty: {reading_difficulty}\nSentiment: {sentiment}\nArousal: {arousal}\n")

            # Write to file
            file.write(
                f"Sentence: {sentence}\nDFU Coefficient: {dfu_coefficient}\nReading Difficulty: {reading_difficulty}\nSentiment: {sentiment}\nArousal: {arousal}\n\n")

            # Store results in lists:

            # Measure of interest:
            reading_material_DFU.append(dfu_coefficient)  # (0='neutral' and 1='de facto unfalsifiable')

            # Confounding variables
            reading_material_DIF.append(
                round(reading_difficulty, 3))  # (0='extremely easy' to 100='extremely difficult')
            reading_material_SEN.append(sentiment)  # (-10='negative', 0='neutral' and 10='positive')
            reading_material_ARO.append(arousal)  # (0='low arousal' and 10='high arousal')


# Calculate metrics for reading material
output_file = "reading_material_metrics.txt"
calculate_4_metrics(reading_material, output_file)


# Compute statistical measures
def compute_statistics(data):
    return {
        "mean": np.mean(data),
        "std_dev": np.std(data),
        "range": (np.min(data), np.max(data)),
        "median": np.median(data),
    }


# Compute statistics for each metric
dfu_stats = compute_statistics(reading_material_DFU)
dif_stats = compute_statistics(reading_material_DIF)
sen_stats = compute_statistics(reading_material_SEN)
aro_stats = compute_statistics(reading_material_ARO)

# Write statistical summaries to the file
with open(output_file, "a") as file:
    file.write("DFU Statistics:\n" + str(dfu_stats) + "\n")
    file.write("Reading Difficulty Statistics:\n" + str(dif_stats) + "\n")
    file.write("Sentiment Statistics:\n" + str(sen_stats) + "\n")
    file.write("Arousal Statistics:\n" + str(aro_stats) + "\n")

# Print to console
print("DFU Statistics:", dfu_stats)
print("Reading Difficulty Statistics:", dif_stats)
print("Sentiment Statistics:", sen_stats)
print("Arousal Statistics:", aro_stats)
