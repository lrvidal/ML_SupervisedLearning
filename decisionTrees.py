from FileParser import FileParser
import Utils as utils
import numpy as np
import math

class DecisionTrees:
    def __init__(self):
        self.tree = None

    def fit(self, data):
        self.tree = self.build_tree(data)

    def build_tree(self, data, depth=0, min_samples_split=2, max_depth=5, min_info_gain=0.01):
        if all(target == data[0][-1] for target in data):
            return data[0][-1]

        if depth == max_depth:
            return self.most_common_label(data)

        if len(data) < min_samples_split:
            return self.most_common_label(data)

        best_feature = self.find_best_feature(data)

        if self.calculate_info_gain(data, best_feature) < min_info_gain:
            return self.most_common_label(data)

        left_split, right_split = self.split_data(data, best_feature)

        left_subtree = self.build_tree(left_split, depth + 1)
        right_subtree = self.build_tree(right_split, depth + 1)

        return (best_feature, left_subtree, right_subtree)

    def most_common_label(self, data):
        labels = [row[-1] for row in data]
        return max(set(labels), key=labels.count)

    def find_best_feature(self, data):
        best_feature = None
        best_info_gain = -1

        num_features = len(data[0]) - 1  # The last column is the target variable

        if all(target == data[0][-1] for target in data):
            return None


        for feature in range(num_features):
            info_gain = self.calculate_info_gain(data, feature)

            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature

        return best_feature

    def calculate_info_gain(self, data, feature):
        entropy_before = self.calculate_entropy(data)

        left_split, right_split = self.split_data(data, feature)

        entropy_left = self.calculate_entropy(left_split)
        entropy_right = self.calculate_entropy(right_split)

        entropy_after = len(left_split) / len(data) * entropy_left + len(right_split) / len(data) * entropy_right

        return entropy_before - entropy_after

    def calculate_entropy(self, data):
        class_counts = {}
        for row in data:
            class_label = row[-1]
            if class_label not in class_counts:
                class_counts[class_label] = 0
            class_counts[class_label] += 1

        entropy = 0
        total_count = len(data)
        for class_label in class_counts:
            proportion = class_counts[class_label] / total_count
            entropy -= proportion * math.log2(proportion)

        return entropy

    def split_data(self, data, feature):
        true_data = [row for row in data if row[feature]]
        false_data = [row for row in data if not row[feature]]
        return true_data, false_data
    
    def predict(self, input_data):
        predictions = [self.traverse_tree(data, self.tree) for data in input_data]
        return predictions

    def traverse_tree(self, data, subtree):
        if not isinstance(subtree, tuple):
            return subtree

        feature, left_subtree, right_subtree = subtree

        if data[feature]:
            return self.traverse_tree(data, left_subtree)
        else:
            return self.traverse_tree(data, right_subtree)


