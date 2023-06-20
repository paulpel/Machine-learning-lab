import json
import glob

# Lista plików JSON
json_files = glob.glob('results/average_*.json')

# Przechowywanie wyników
ranking_results = {}

# Pętla przez wszystkie pliki JSON
for json_file in json_files:
    with open(json_file) as file:
        json_data = json.load(file)
        
    # Przechodzenie przez wszystkie datasety w pliku JSON
    for dataset_name, dataset in json_data.items():
        # Pobieranie wyników dla klasyfikatorów
        random_forest_result = dataset['random_forest']['balanced_accuracy']
        decision_tree_result = dataset['decision_tree']['balanced_accuracy']
        naive_bayes_result = dataset['naive_bayes']['balanced_accuracy']
        
        # Wykonanie testu rankingowego na podstawie wyników balanced_accuracy
        ranking = sorted([
            ('Random Forest', random_forest_result),
            ('Decision Tree', decision_tree_result),
            ('Naive Bayes', naive_bayes_result)
        ], key=lambda x: x[1], reverse=True)
        
        # Dodanie wyników do słownika ranking_results
        if dataset_name not in ranking_results:
            ranking_results[dataset_name] = []
        
        ranking_results[dataset_name].append(ranking)
        
# Wyświetlanie wyników testu rankingowego dla każdego datasetu
for dataset_name, rankings in ranking_results.items():
    print(f"Dataset: {dataset_name}")
    for i, ranking in enumerate(rankings):
        print(f"Ranking {i + 1}:")
        for classifier, score in ranking:
            print(f"{classifier}: {score}")
        print("--------------------")
