import csv
import math
import random


def load_data(filename):
    data = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header if exists
        for row in reader:
            # Adjust indexing depending on actual data columns
            # Assume last column is "diagnosis" 'M' or 'B'
            # and the rest are features.
            label_str = row[1].strip()
            label = 1.0 if label_str == "M" else 0.0
            # Suppose features start at column 2 until the end
            # This depends on how your CSV is structured. Check the dataset!
            features = list(map(float, row[2:]))
            data.append((features, label))

    random.shuffle(data)
    return data


def normalize_data(train_data, test_data):
    # Compute mean and std for each feature
    num_features = len(train_data[0][0])
    means = [
        sum(sample[0][i] for sample in train_data) / len(train_data)
        for i in range(num_features)
    ]
    stds = []
    for i in range(num_features):
        variance = sum((sample[0][i] - means[i]) ** 2 for sample in train_data) / len(
            train_data
        )
        stds.append(math.sqrt(variance))

    def normalize(sample):
        return [
            (sample[i] - means[i]) / stds[i] if stds[i] != 0 else sample[i]
            for i in range(num_features)
        ]

    train_data = [(normalize(f), l) for f, l in train_data]
    test_data = [(normalize(f), l) for f, l in test_data]

    return train_data, test_data


def get_train_test_data(filename, train_ratio=0.8):
    data = load_data(filename)
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    train_data, test_data = normalize_data(train_data, test_data)
    return train_data, test_data
