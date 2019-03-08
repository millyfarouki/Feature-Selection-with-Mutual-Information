import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import dit
import string
import seaborn as sns
import time


# from sklearn import metrics
# import cProfile


class Parameters:
    """Ensure that N_FEATURES_SELECTED is less than N_WORKING_FEATURES"""
    N_WORKING_FEATURES = 9
    N_WORKING_SAMPLES = 10
    N_BINS = 20
    N_FEATURES_SELECTED = 3
    ALPHA = 1
    BETA = 1


class MutualInformationRepository:
    def __init__(self, n_features):
        self.relevancy = [0] * n_features
        self.redundancy = np.zeros((n_features, n_features))
        self.redundancy_acceptable = np.zeros((n_features, n_features))


def matrix_print(mat, fmt="g"):
    if isinstance(mat, list): mat = np.array(mat)
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")


def open_csv_file(data_path):
    with open(data_path, newline='') as csvfile:
        reader = list(csv.reader(csvfile, delimiter=','))
        del reader[0]
        for line in reader:
            del line[0]
        data_array = reader
        return data_array


def cast_as_floats(list_of_lists):
    float_list = [[float(item) for item in sublist] for sublist in list_of_lists]
    return float_list


def single_list(list_of_lists):
    flat_list = []
    for sublist in list_of_lists:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def working_subset(data_matrix, gene_labels, n_samples, n_features):
    data_subset = []
    if n_samples == 0: n_samples = len(data_matrix)
    if n_features == 0: n_features = len(data_matrix[0])
    if n_samples > len(data_matrix): n_samples = len(data_matrix)
    for gene_number in range(n_samples):
        gene_row = data_matrix[gene_number]
        data_subset.append(gene_row[0:n_features])
    label_subset = gene_labels[0:n_samples]
    return data_subset, label_subset


def load_data(n_samples, n_features):
    data_file = open_csv_file('data/data.csv')
    data_file = cast_as_floats(data_file)
    labels_file = open_csv_file('data/labels.csv')
    labels_file = single_list(labels_file)
    (data_subset, label_subset) = working_subset(data_file, labels_file, n_samples, n_features)
    return data_subset, label_subset


def find_unique_classifiers(label_list):
    unique_labels = list(set(label_list))
    return unique_labels


def bin_data(data_matrix, n_bins):
    lower_limit = np.floor(min(min(data_matrix)))
    upper_limit = np.ceil(max(max(data_matrix)))
    bins = np.linspace(int(lower_limit), int(upper_limit), n_bins + 1)
    digitised_data = np.digitize(data_matrix, bins)
    return digitised_data


def remove_feature_list_overlap(features_selected, remaining_features):
    remaining_features = set(remaining_features) - set(features_selected)
    return list(remaining_features)


def rescale_data(data_values):
    df = pd.DataFrame(data_values)
    scaled_data = StandardScaler().fit_transform(df)
    return scaled_data


def remove_zero_lines(target_matrix: np.ndarray):
    retained_indices = np.argwhere(~(target_matrix == 0).all(0))
    retained_indices = single_list(retained_indices)
    reduced_matrix_rows = target_matrix[~(target_matrix == 0).all(0)]
    reduced_matrix_rows = reduced_matrix_rows.transpose()  # type: np.ndarray
    reduced_matrix = reduced_matrix_rows[~(reduced_matrix_rows == 0).all(1)]
    return reduced_matrix, retained_indices


def remove_nan_lines(target_matrix):
    retained_indices = np.argwhere(~(np.isnan(target_matrix)).all(0))
    retained_indices = single_list(retained_indices)
    reduced_matrix_rows = target_matrix[~(np.isnan(target_matrix)).all(0)]
    reduced_matrix_rows = reduced_matrix_rows.transpose()
    reduced_matrix = reduced_matrix_rows[~(np.isnan(reduced_matrix_rows)).all(1)]
    return reduced_matrix, retained_indices


def check_input_reasonability(n_features, max_features):
    if max_features >= n_features:
        print('Number of features selected cannot exceed number of features in dataset. Check input parameters.')
        exit()


def prepare_data_for_mi(working_data, working_labels):
    n_features = len(working_data[0])
    features_selected = [random.randint(0, n_features - 1)]
    remaining_features = remove_feature_list_overlap(features_selected, range(n_features))
    binned_data = bin_data(working_data, Parameters.N_BINS)
    all_labels = find_unique_classifiers(working_labels)
    numeric_labels = assign_arbitrary_values(all_labels, working_labels)
    working_data = np.array(binned_data)
    mi_stored_values = MutualInformationRepository(n_features)
    return working_data, numeric_labels, features_selected, remaining_features, mi_stored_values


def assign_arbitrary_values(unique_label_list, full_label_list):
    value_dictionary = {}
    key_values = [[i] for i in range(1, len(unique_label_list) + 1)]
    random.shuffle(key_values)
    key_values = single_list(key_values)
    i = 0
    for value in key_values:
        value_dictionary[unique_label_list[i]] = value
        i += 1
    reassigned_labels = [value_dictionary.get(item, item) for item in full_label_list]
    return reassigned_labels


def make_dit_distribution(distribution_matrix, columns,
                          *classifier_columns):  # (distribution_matrix, columns, classifier_column):
    rv_names = make_dit_rv_names(len(columns) + len(classifier_columns))
    sample_matrix = distribution_matrix[:, np.r_[columns]]
    full_matrix = []
    if len(classifier_columns) == 0:
        full_matrix = sample_matrix
    elif len(classifier_columns) == 1:
        classifier_column = classifier_columns[0]
        classifier_column_array = np.array([classifier_column])
        try:
            full_matrix = np.c_[sample_matrix, classifier_column_array]
        except ValueError:
            full_matrix = np.c_[sample_matrix, classifier_column_array.transpose()]
    else:
        print('Too many classifier columns have been passed to make a dit distribution')
        exit()
    dit_distribution = build_dit_structure(full_matrix, rv_names)
    return dit_distribution


def build_dit_structure(full_matrix, rv_names):
    n_samples = len(full_matrix)
    outcomes = []
    for row in full_matrix:
        alpha_entry = alphabetise_data(row.tolist())
        outcomes.append(alpha_entry)
    pmf = [1 / n_samples] * n_samples
    dit_distribution = dit.Distribution(outcomes, pmf)
    dit_distribution.set_rv_names(rv_names)
    return dit_distribution


def alphabetise_data(data_row):
    alpha_list = [chr(item + 96) for item in data_row]
    alpha_string = ''.join(alpha_list)
    return alpha_string


def make_dit_rv_names(n_variables):
    all_letters = string.ascii_uppercase
    variables = all_letters[-n_variables:]
    return list(variables)


def n_features_check(n_features_to_show, input_matrix):
    if n_features_to_show > input_matrix.shape[1]:
        n_features_to_show = input_matrix.shape[1]
    return n_features_to_show


def heatmap(array, *axis_labels):
    mask = np.zeros_like(array)
    mask[np.triu_indices_from(mask, 1)] = True
    with sns.axes_style("white"):
        if len(axis_labels) > 0:
            axis_labels = axis_labels[0]
            sns.heatmap(array, linewidth=0.5, mask=mask, square=True, xticklabels=axis_labels, yticklabels=axis_labels)
        else:
            sns.heatmap(array, linewidth=0.5, mask=mask, square=True)
    plt.show()


def prepare_pca_data(n_samples, n_features):
    start_time = time.time()
    (working_data, working_labels) = load_data(n_samples, n_features)
    scaled_data = rescale_data(working_data)
    print("--- data prepared for pca in %s seconds ---" % (time.time() - start_time))
    return working_data, scaled_data, working_labels


def prepare_mi_data(working_data, working_labels):
    start_time = time.time()
    (binned_data, numeric_labels, features_selected, remaining_features, mi_stored_values) = prepare_data_for_mi(
        working_data, working_labels)
    print("--- data prepared for mutual information in %s seconds ---" % (time.time() - start_time))

    return binned_data, numeric_labels, features_selected, remaining_features, mi_stored_values


def initialise_remaining_features_correlation(n_features, starting_feature):
    remaining_features = list(range(n_features))
    remaining_features.remove(starting_feature)
    return remaining_features


def calculate_relevancy(data_matrix, label_vector, test_feature, mi_stored_values):
    if mi_stored_values.relevancy[test_feature] == 0:
        dit_distribution = make_dit_distribution(data_matrix, [test_feature], label_vector)
        mutual_information = dit.shannon.mutual_information(dit_distribution, ['Y'], ['Z'])
        mi_stored_values.relevancy[test_feature] = mutual_information
    return mi_stored_values.relevancy[test_feature]


def populate_redundancy(data_matrix, feature_1, feature_2, mi_stored_values):
    if mi_stored_values.redundancy[feature_1, feature_2] != 0:
        mutual_information = mi_stored_values.redundancy[feature_1, feature_2]
    else:
        test_columns = [feature_1, feature_2]
        dit_distribution = make_dit_distribution(data_matrix, test_columns)
        mutual_information = dit.shannon.mutual_information(dit_distribution, ['Y'], ['Z'])
        mi_stored_values.redundancy[feature_1, feature_2] = mutual_information
        mi_stored_values.redundancy[feature_2, feature_1] = mutual_information
        if mutual_information < 0:
            print('mutual information is', mutual_information, 'for previous feature', feature_1,
                  'and test feature', feature_2)
            print(dit_distribution)
    return mutual_information


def population_redundancy_acceptable(data_matrix, previous_feature, test_feature, label_vector, mi_stored_values):
    if mi_stored_values.redundancy_acceptable[previous_feature, test_feature] != 0:
        acceptable_mi = mi_stored_values.redundancy_acceptable[previous_feature, test_feature]
    else:
        test_columns = [test_feature, previous_feature]
        dit_distribution = make_dit_distribution(data_matrix, test_columns, label_vector)
        joint_mi = dit.shannon.mutual_information(dit_distribution, ['Y'], ['X', 'Z'])
        acceptable_mi = 0
        if mi_stored_values.relevancy[previous_feature] == 0:
            previous_relevancy = dit.shannon.mutual_information(dit_distribution, ['Y'], ['Z'])
            mi_stored_values.relevancy[previous_feature] = previous_relevancy
        else:
            previous_relevancy = mi_stored_values.relevancy[previous_feature]
            acceptable_mi = joint_mi - previous_relevancy
            mi_stored_values.redundancy_acceptable[previous_feature, test_feature] = acceptable_mi
            mi_stored_values.redundancy_acceptable[test_feature, previous_feature] = acceptable_mi
    return acceptable_mi


def calculate_redundancy(data_matrix, test_feature, features_selected, mi_stored_values):
    mutual_information_sum = 0
    for previous_feature in features_selected:
        mutual_information = populate_redundancy(data_matrix, previous_feature, test_feature, mi_stored_values)
        mutual_information_sum += mutual_information
    return mutual_information_sum


def calculate_acceptable_redundancy(data_matrix, label_vector, test_feature, features_selected, mi_stored_values):
    mutual_information_sum = 0
    for previous_feature in features_selected:
        acceptable_mi = population_redundancy_acceptable(data_matrix, previous_feature, test_feature, label_vector, mi_stored_values)
        mutual_information_sum += acceptable_mi
    return mutual_information_sum


def calculate_redundancy_correlation(correlation_matrix, test_feature, features_selected):
    correlation_sum = 0
    for previous_feature in features_selected:
        correlation_sum += correlation_matrix[previous_feature, test_feature]
    return correlation_sum


def select_next_feature_mi(binned_data, numeric_labels, features_selected, remaining_features, mi_stored_values):
    arg_max = 0
    new_feature_selected = remaining_features[0]
    i = 1
    for feature_number in remaining_features:
        feature_selection_value = test_feature_mi(feature_number, binned_data, numeric_labels, features_selected,
                                                  mi_stored_values)
        if feature_selection_value > arg_max:
            arg_max = feature_selection_value
            new_feature_selected = feature_number
        i += 1
    return new_feature_selected


def select_next_feature_correlation(features_selected, remaining_features, correlation_matrix):
    arg_max = 0
    new_feature_selected = remaining_features[0]
    i = 1
    for feature_number in remaining_features:
        feature_selection_value = test_feature_correlation(feature_number, features_selected, correlation_matrix)
        if feature_selection_value > arg_max:
            arg_max = feature_selection_value
            new_feature_selected = feature_number
        i += 1
    return new_feature_selected


def test_feature_mi(feature_number, data_matrix, label_vector, features_selected, mi_stored_values):
    if len(features_selected) <= 1:
        n_features_selected = 2
    else:
        n_features_selected = len(features_selected)

    relevancy_mi = calculate_relevancy(data_matrix, label_vector, feature_number, mi_stored_values)
    redundancy = calculate_redundancy(data_matrix, feature_number, features_selected, mi_stored_values)
    redundancy_acceptable = calculate_acceptable_redundancy(data_matrix, label_vector, feature_number,
                                                            features_selected, mi_stored_values)

    alpha = Parameters.ALPHA * (1 / (1 - n_features_selected))
    beta = Parameters.BETA * (1 / (1 - n_features_selected))

    return relevancy_mi - (alpha * redundancy - beta * redundancy_acceptable)


def test_feature_correlation(feature_number, features_selected, correlation_matrix):
    if len(features_selected) <= 1:
        n_features_selected = 2
    else:
        n_features_selected = len(features_selected)

    relevancy = correlation_matrix[feature_number, -1]
    redundancy = calculate_redundancy_correlation(correlation_matrix, feature_number, features_selected)

    alpha = Parameters.ALPHA * (1 / (1 - n_features_selected))

    return relevancy - (alpha * redundancy)


def select_features_with_pca(data_values, labels, n_features):
    start_time = time.time()
    pca_model = PCA(n_components=n_features)
    principle_components = pca_model.fit_transform(data_values)
    variance_accounted_for = pca_model.explained_variance_ratio_
    print(n_features, 'out of', Parameters.N_WORKING_FEATURES, 'features can account for',
          sum(variance_accounted_for), 'of the total variance')
    if n_features == 2:
        visualise_pca(principle_components, labels)
    print("--- PCA performed in %s seconds ---" % (time.time() - start_time))
    return principle_components


def select_features_with_mi(binned_data, numeric_labels, features_selected, remaining_features, max_features,
                            mi_stored_values):
    while len(features_selected) < max_features:
        print(len(features_selected), 'feature(s) currently selected using mutual information')
        start_time = time.time()
        new_feature = select_next_feature_mi(binned_data, numeric_labels, features_selected, remaining_features,
                                             mi_stored_values)
        features_selected.append(new_feature)
        remaining_features = remove_feature_list_overlap(features_selected, remaining_features)
        print("--- %s seconds to select next feature ---" % (time.time() - start_time))
    return features_selected


def select_features_with_correlation(scaled_data, numeric_labels, features_selected, remaining_features, max_features,
                                     n_features_to_show=20):
    label_column = np.array([numeric_labels])
    combined_array = np.append(scaled_data, label_column.transpose(), axis=1)
    correlation_matrix = np.corrcoef(combined_array, rowvar=False)
    visualise_correlation(correlation_matrix, n_features_to_show)
    features_selected = [features_selected]
    while len(features_selected) < max_features:
        print(len(features_selected), 'feature(s) currently selected using correlation')
        start_time = time.time()
        new_feature = select_next_feature_correlation(features_selected, remaining_features, correlation_matrix)
        features_selected.append(new_feature)
        remaining_features = remove_feature_list_overlap(features_selected, remaining_features)
        print("--- %s seconds to select next feature ---" % (time.time() - start_time))
    return features_selected


def visualise_mi_matrix(mi_matrix, redundancy_type, data_matrix, n_features_to_show, mi_stored_values, label_vector):
    n_features_to_show = n_features_check(n_features_to_show, mi_matrix)
    for feature_1 in range(n_features_to_show):
        for feature_2 in range(feature_1, n_features_to_show):
            if redundancy_type == 'unacceptable':
                populate_redundancy(data_matrix, feature_1, feature_2, mi_stored_values)
            elif redundancy_type == 'acceptable':
                population_redundancy_acceptable(data_matrix, feature_1, feature_2, label_vector, mi_stored_values)
    (reduced_matrix, retained_lines) = remove_zero_lines(mi_matrix)
    heatmap(reduced_matrix, retained_lines)
    return reduced_matrix


def visualise_pca(pca_components, data_labels):
    x_limit = np.ceil(max(abs(pca_components[:, 0]))); y_limit = np.ceil(max(abs(pca_components[:, 1])))
    unique_labels = find_unique_classifiers(data_labels)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('2 component PCA', fontsize=20)
    ax.set_xlabel('Principal Component 1', fontsize=15); ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_xlim([-x_limit, x_limit]); ax.set_ylim([-y_limit, y_limit])

    for label in unique_labels:
        indices = [i for i, x in enumerate(data_labels) if x == label]
        ax.scatter(pca_components[indices, 0], pca_components[indices, 1])

    ax.grid(); ax.legend(unique_labels)
    plt.show()


def visualise_mi(data_matrix, label_vector, mi_stored_values, n_features_to_show=20):
    standard_arguments = [data_matrix, n_features_to_show, mi_stored_values, label_vector]
    redundancy = visualise_mi_matrix(mi_stored_values.redundancy, 'unacceptable', *standard_arguments)
    redundancy_acceptable = visualise_mi_matrix(mi_stored_values.redundancy_acceptable, 'acceptable', *standard_arguments)
    overall_redundancy = redundancy - redundancy_acceptable
    visualise_mi_matrix(overall_redundancy, 'neither', *standard_arguments)


def visualise_correlation(correlation_matrix, n_features_to_show):
    n_features_to_show = n_features_check(n_features_to_show - 1, correlation_matrix)
    (reduced_matrix, retained_lines) = remove_nan_lines(correlation_matrix[0:n_features_to_show, 0:n_features_to_show])
    heatmap(reduced_matrix, retained_lines)
    plt.show()


def main_loop(n_samples=Parameters.N_WORKING_SAMPLES, n_features=Parameters.N_WORKING_FEATURES,
              max_features=Parameters.N_FEATURES_SELECTED):
    check_input_reasonability(n_features, max_features)
    (working_data, scaled_data, working_labels) = prepare_pca_data(n_samples, n_features)
    # pca_components = select_features_with_pca(scaled_data, working_labels, max_features)
    (binned_data, numeric_labels, features_selected_mi, remaining_features_mi, mi_stored_values) = prepare_mi_data(
        working_data, working_labels)
    features_selected_mi = select_features_with_mi(binned_data, numeric_labels, features_selected_mi,
                                                   remaining_features_mi, max_features, mi_stored_values)
    visualise_mi(binned_data, numeric_labels, mi_stored_values)
    print('Features selected using mutual information are', features_selected_mi)
    remaining_features_correlation = initialise_remaining_features_correlation(len(scaled_data),
                                                                               features_selected_mi[0])
    features_selected_correlation = select_features_with_correlation(scaled_data, numeric_labels,
                                                                     features_selected_mi[0],
                                                                     remaining_features_correlation, max_features)
    print('Features selected using correlation are', features_selected_correlation)


# TODO: Look at alternative ways of dealing with continuous variables
# TODO: Visualise mutual information relevancy
# TODO: Estimate mutual information for speed gain
# cProfile.run('main_loop()')
main_loop()
