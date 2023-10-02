import math
import pandas as pd

def euclidean_distance(row1, row2):
    if len(row1) != len(row2):
        return None
    ret_val = 0
    for idx, item in enumerate(row1):
        ret_val += (row1[idx] - row2[idx]) ** 2
    return math.sqrt(ret_val)

def get_accuracy(true_output, predicted_output):
    correct = 0
    for idx, outcome in enumerate(true_output):
        if predicted_output[idx] == outcome:
            correct += 1
    return correct / len(true_output) * 100

dataset_file = "./dataset.csv"
mark_records = []

min_marks = {
    "Assignment-1": 100,
    "Assignment-2": 100,
    "Assignment-3": 100,
    "Assignment-4": 100,
    "Assignment-5": 100,
    "Final": 100,
    "Mid": 100,
}

max_marks = {
    "Assignment-1": -100,
    "Assignment-2": -100,
    "Assignment-3": -100,
    "Assignment-4": -100,
    "Assignment-5": -100,
    "Final": -100,
    "Mid": -100,
}

def load_min_max(df):
    for k in min_marks:
        min_marks[k] =  min(df.loc[:, k])
        max_marks[k] = max(df.loc[:, k])

def get_normalized_entry(row):
    result = []
    result.append((row[0] - min_marks["Assignment-1"]) / (max_marks["Assignment-1"] - min_marks["Assignment-1"]))
    result.append((row[1] - min_marks["Assignment-2"]) / (max_marks["Assignment-2"] - min_marks["Assignment-2"]))
    result.append((row[2] - min_marks["Assignment-3"]) / (max_marks["Assignment-3"] - min_marks["Assignment-3"]))
    result.append((row[3] - min_marks["Assignment-4"]) / (max_marks["Assignment-4"] - min_marks["Assignment-4"]))
    result.append((row[4] - min_marks["Assignment-5"]) / (max_marks["Assignment-5"] - min_marks["Assignment-5"]))
    result.append((row[5] - min_marks["Final"]) / (max_marks["Final"] - min_marks["Final"]))
    result.append((row[6] - min_marks["Mid"]) / (max_marks["Mid"] - min_marks["Mid"]))
    return result

df = pd.read_csv(dataset_file)
load_min_max(df)

for row in range(len(df)):
    current_record = list(df.loc[row, :])
    current_record_updated = [current_record[0]]
    current_record_updated.extend(get_normalized_entry(current_record[1: len(current_record) - 1]))
    current_record_updated.append(current_record[-1])
    mark_records.append(current_record_updated)

def knn(training_data, validation_data, k):
    predicted_output = []
    for entry in validation_data:
        dist_vector = []
        for compare_entry in training_data:
            if entry != compare_entry:
                sample1 = entry[1: len(entry) - 1]
                sample2 = compare_entry[1: len(compare_entry) - 1]
                dist_vector.append((euclidean_distance(sample1, sample2), compare_entry[-1]))

        dist_vector.sort(key=lambda x: x[0])
        # for k = 1
        nearest_neighbors = [x[1] for x in dist_vector[:k]]
        predicted_output.append(max(set(nearest_neighbors), key=nearest_neighbors.count))

    true_output = [entry[-1] for entry in validation_data]
    accuracy = get_accuracy(true_output, predicted_output)
    return accuracy

training = mark_records[: math.floor(len(mark_records)*0.8)]
validation = mark_records[math.floor(len(mark_records)*0.8): math.floor(len(mark_records)*0.8) + math.floor(len(mark_records)*0.1)]

k = 1
accuracy = knn(training, validation, k)
print(f"Accuracy on the validation set with k={k}: {accuracy}")
