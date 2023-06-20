import json
import glob

# json list
json_files = glob.glob('results/average_*.json')

# store results
ranking_results = {}

# looping through jsons
for json_file in json_files:
    with open(json_file) as file:
        json_data = json.load(file)
        
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
        
        ranking_results[dataset_name].append(ranking)
        
# Display ranking test results for each dataset
for dataset_name, rankings in ranking_results.items():
    print(f"Dataset: {dataset_name}")
    for i, ranking in enumerate(rankings):
        print(f"Ranking {i + 1}:")
        for classifier, score in ranking:
            print(f"{classifier}: {score}")
        print("--------------------")
