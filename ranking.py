import json
import glob
import pandas as pd

# List of JSON files
json_files = glob.glob('results/average_*.json')

# Store results
ranking_results = {}

# Loop through all JSON files
for json_file in json_files:
    with open(json_file) as file:
        json_data = json.load(file)

    # Extract the filename without the extension
    json_filename = json_file.split('/')[-1].split('.')[0]

    # Iterate through all datasets in the JSON file
    for dataset_name, dataset in json_data.items():
        # Get results for classifiers
        random_forest_result = dataset['random_forest']['balanced_accuracy']
        decision_tree_result = dataset['decision_tree']['balanced_accuracy']
        naive_bayes_result = dataset['naive_bayes']['balanced_accuracy']

        # Perform ranking test based on balanced_accuracy results
        ranking = sorted([
            ('Random Forest', random_forest_result),
            ('Decision Tree', decision_tree_result),
            ('Naive Bayes', naive_bayes_result)
        ], key=lambda x: x[1], reverse=True)

        # Add results to ranking_results dictionary
        if dataset_name not in ranking_results:
            ranking_results[dataset_name] = []

        ranking_results[dataset_name].append((json_filename, ranking))

df = pd.DataFrame(columns=['Dataset', 'Method', 'Classifier', 'Score'])

# Display ranking test results for each dataset
for dataset_name, rankings in ranking_results.items():
    for i, (json_filename, ranking) in enumerate(rankings):
        for classifier, score in ranking:
            temp_df = pd.DataFrame({'Dataset': dataset_name, 'Method': json_filename.split("_")[-1], 'Classifier': classifier, 'Score': score}, index=[0])
            df = pd.concat([df, temp_df], ignore_index=True)

# Print the resulting DataFrame
print(df)
df.to_excel('ranking_results.xlsx', index=False)

