import sys
import pandas as pd
import numpy as np
import time

from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from sklearn.tree import DecisionTreeClassifier, plot_tree
import operator

# from https://www.python-course.eu/Decision_Trees.php


def entropy(target_col):
    """
    Calculate the entropy of a dataset.
    The only parameter of this function is the target_col parameter which specifies the target column
    """

    counts, elements = np.histogram(target_col.dropna())
    entropy = 0
    for i in range(len(elements)-1):
        if (counts[i] != 0):
            entropy += (-counts[i]/np.sum(counts)) * \
                np.log2(counts[i] / np.sum(counts))

    return entropy


def InfoGain(data, split_attribute_name, target_name="Class"):
    """
    Calculate the information gain of a dataset. This function takes three parameters:
    1. data = The dataset for whose feature the IG should be calculated
    2. split_attribute_name = the name of the feature for which the information gain should be calculated
    3. target_name = the name of the target feature. The default for this example is "class"
    """
    data[split_attribute_name] = pd.to_numeric(data[split_attribute_name])
    # Calculate the values and the corresponding counts for the split attribute
    counts, vals = np.histogram(data[split_attribute_name].dropna())

    # Calculate the weighted entropy
    Weighted_Entropy = 0
    for i in range(len(vals)-1):
        if (counts[i] != 0):
            Weighted_Entropy += (counts[i]/np.sum(counts))*entropy(data.where(
                (data[split_attribute_name] < vals[i+1]) & (data[split_attribute_name] >= vals[i])).dropna()[target_name])

    return Weighted_Entropy


def get_continuous_attributes(name_dataset, data):
    """
    Returns the continuous attributes
    1. name_var = The name of the dataset
    2. data = the dataset
    """
    continuous_attributes = []

    if name_dataset == "aps_failure":
        continuous_attributes = data.columns[1:]
    elif name_dataset == "colic":
        continuous_attributes = data.columns
    elif name_dataset == "spambase":
        continuous_attributes = data.columns
    elif name_dataset == "hydraulic":
        continuous_attributes = data.columns
    elif name_dataset == "hydraulic_stable":
        continuous_attributes = data.columns
    elif name_dataset == "hydraulic_cooler":
        continuous_attributes = data.columns
    elif name_dataset == "hydraulic_valve":
        continuous_attributes = data.columns
    elif name_dataset == "hydraulic_leakage":
        continuous_attributes = data.columns
    elif name_dataset == "hydraulic_accumulator":
        continuous_attributes = data.columns
    elif name_dataset == "ionosphere":
        continuous_attributes = data.columns[2:]
    elif name_dataset == "heart-statlog":
        continuous_attributes = ["age", "restingBloodPpressure", "serumCholestoral",
                                 "maximumHeartRateAchieved", "oldpeak", "slopePeakExercise", "numOfMajorVessels"]
    # elif name_dataset == "heart-statlog":
    #     continuous_attributes = data.columns

    return continuous_attributes


def nb_att_1(name_dataset):
    """
    Returns the number of variables in A submodel
    1. name_var = The name of the dataset
    """
    nb_att = 0

    if name_dataset == "aps_failure":
        nb_att = 85
    elif name_dataset == "colic":
        nb_att = 5
    elif name_dataset == "spambase":
        nb_att = 10
    elif name_dataset == "hydraulic":
        nb_att = 11
    elif name_dataset == "hydraulic_stable":
        nb_att = 8
    elif name_dataset == "ionosphere":
        nb_att = 9
    elif name_dataset == "heart-statlog":
        nb_att = 4

    return nb_att


def nb_att_2(name_dataset):
    """
    Returns the number of variables in A submodel
    1. name_var = The name of the dataset
    """
    nb_att = 0

    if name_dataset == "aps_failure":
        nb_att = 85
    elif name_dataset == "colic":
        nb_att = 4
    elif name_dataset == "spambase":
        nb_att = 10
    elif name_dataset == "hydraulic":
        nb_att = 10
    elif name_dataset == "hydraulic_stable":
        nb_att = 7
    elif name_dataset == "ionosphere":
        nb_att = 9
    elif name_dataset == "heart-statlog":
        nb_att = 3

    return nb_att


def export_csv(name_dataset, data, labels, name_method):
    # over-sampling
    resampler = ADASYN(random_state=42)
    data_resampled, labels_resampled = resampler.fit_resample(data, labels)

    carpet = "../MO2P/data/all/"

    print("generate")
    full_data = carpet + "./" + name_dataset + "_data_train_" + \
        name_method + "_" + str(nb_att_1) + "_" + str(nb_att_2) + ".csv"
    data_A = carpet + "./" + name_dataset + "_data_train_" + \
        name_method + "_A_" + str(nb_att_1) + "_" + str(nb_att_2) + ".csv"
    data_B = carpet + "./" + name_dataset + "_data_train_" + \
        name_method + "_B_" + str(nb_att_1) + "_" + str(nb_att_2) + ".csv"
    labels_A = carpet + "./" + name_dataset + "_truth_train_" + \
        name_method + "_A_" + str(nb_att_1) + "_" + str(nb_att_2) + ".csv"
    labels_B = carpet + "./" + name_dataset + "_truth_train_" + \
        name_method + "_B_" + str(nb_att_1) + "_" + str(nb_att_2) + ".csv"

    full_data_resampled = carpet + "./" + name_dataset + "_data_train_" + \
        name_method + "_ADASYN_" + str(nb_att_1) + "_" + str(nb_att_2) + ".csv"
    data_A_resampled = carpet + "./" + name_dataset + "_data_train_" + \
        name_method + "_A_ADASYN_" + \
        str(nb_att_1) + "_" + str(nb_att_2) + ".csv"
    data_B_resampled = carpet + "./" + name_dataset + "_data_train_" + \
        name_method + "_B_ADASYN_" + \
        str(nb_att_1) + "_" + str(nb_att_2) + ".csv"
    labels_A_resampled = carpet + "./" + name_dataset + "_truth_train_" + \
        name_method + "_A_ADASYN_" + \
        str(nb_att_1) + "_" + str(nb_att_2) + ".csv"
    labels_B_resampled = carpet + "./" + name_dataset + "_truth_train_" + \
        name_method + "_B_ADASYN_" + \
        str(nb_att_1) + "_" + str(nb_att_2) + ".csv"

    print(full_data)
    print(data_A)
    print(data_B)
    print(labels_A)
    print(labels_B)

    print(full_data_resampled)
    print(data_A_resampled)
    print(data_B_resampled)
    print(labels_A_resampled)
    print(labels_B_resampled)

    data_resampled                  .to_csv(full_data_resampled, index=False)
    data_resampled.iloc[:, :nb_att_1].to_csv(data_A_resampled, index=False)
    data_resampled.iloc[:, nb_att_1:].to_csv(data_B_resampled, index=False)
    labels_resampled                .to_csv(labels_A_resampled, index=False)
    labels_resampled                .to_csv(labels_B_resampled, index=False)

    data.to_csv(full_data, index=False)
    data.iloc[:, :nb_att_1].to_csv(data_A, index=False)
    data.iloc[:, nb_att_1:].to_csv(data_B, index=False)
    labels.to_csv(labels_A, index=False)
    labels.to_csv(labels_B, index=False)


def split_by_entropy(name_dataset, nb_att_1, nb_att_2):
    continuous_attributes = {}

    filepath = '../UCI_datasets/' + name_dataset + '_treated.csv'

    sep = ","
    data = pd.read_csv(filepath, header=0, sep=sep, engine="python")
    data = data.dropna()

    continuous_attributes = get_continuous_attributes(name_dataset, data)

    # calculate total entropy
    total_entropy = entropy(data["Class"])

    print("---", name_dataset)
    list_gains = {}
    for name_var in continuous_attributes:
        if name_var != "Class":
            gain = total_entropy - InfoGain(data, name_var, "Class")
            list_gains[name_var] = gain
    list_gains = {k: v for k, v in sorted(
        list_gains.items(), key=lambda item: item[1], reverse=True)}

    for i in list_gains.keys():
        print(" ", list_gains[i],  "(", i, ")")

    keys = list(list_gains.keys())
    X_1, X_2 = [], []

    print("view 1")
    for i in range(len(keys)):
        if (i % 2 == 0):
            print(" ", list_gains[keys[i]], "(", keys[i], ")")
            X_1.append(data[keys[i]])
    print("view 2")
    for i in range(len(keys)):
        if (i % 2 == 1):
            print(" ", list_gains[keys[i]], "(", keys[i], ")")
            X_2.append(data[keys[i]])

    print("nb_att_1 " + str(nb_att_1))
    print("nb_att_2 " + str(nb_att_2))

    selected_keys = []
    data_dict = {}
    for i in range(nb_att_1):  # nombre d'attributs left
        print(i*2)
        data_dict[keys[i*2]] = X_1[i]
        selected_keys.append(keys[i*2])
    for i in range(nb_att_2):  # nombre d'attributs right
        print(i*2+1)
        data_dict[keys[i*2+1]] = X_2[i]
        selected_keys.append(keys[i*2+1])
    data_dict["label"] = data["Class"]

    df = pd.DataFrame(data_dict)
    data, labels = df[selected_keys], df["label"]

    return data, labels


def create_dataset(var, nb_att_1, nb_att_2):

    filepath = '../UCI_datasets/' + name_dataset + '_treated.csv'

    sep = ","
    original_data = pd.read_csv(filepath, header=0, sep=sep, engine="python")
    original_data = original_data.dropna()

    var["A"] = var["A"][len(var["A"])-nb_att_1:]
    var["B"] = var["B"][len(var["B"])-nb_att_2:]

    data = original_data[var["A"]]
    data[var["B"]] = original_data[var["B"]]

    labels = original_data["Class"]

    return data, labels


if __name__ == '__main__':
    print(sys.argv[1])
    name_dataset = sys.argv[1]
    start = time.time()

    nb_att_1 = int(sys.argv[2])
    nb_att_2 = int(sys.argv[3])
    data, labels = split_by_entropy(name_dataset, nb_att_1, nb_att_2)
    export_csv(name_dataset, data, labels, "entropy")

    end = time.time()
    print("Needed " + str(end - start) + " seconds")
