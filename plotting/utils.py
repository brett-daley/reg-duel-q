import os
import numpy as np
from scipy.stats import norm

def get_top_level_directories(directory_path):
    if not os.path.exists(directory_path):
        print(f"The specified directory '{directory_path}' does not exist.")
        return []

    # Get a list of all entries in the specified directory
    entries = os.listdir(directory_path)

    # Filter out only directories from the entries
    directories = [entry for entry in entries if os.path.isdir(os.path.join(directory_path, entry))]

    return directories


def find_file(filename, directory):
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    assert False

def insert_space_before_capital(s):
    result = s[0]  # Keep the first character as it is
    for char in s[1:]:
        if char.isupper():
            result += ' ' + char
        else:
            result += char
    return result


def compute_z_score(confidence_level):
    assert 0 < confidence_level < 1, "unsupported confidence level"
    return norm.ppf((1 + confidence_level) / 2)


def interquartile_scores_curve_indices(curves):
    iq_indices = []
    score_curve_y_axes = [curve.values for curve in curves]
    final_scores = [y_axis_val[-1] for y_axis_val in score_curve_y_axes]
    val_map = {i : final_scores[i] for i in range(len(final_scores))}
    q1 = np.percentile(final_scores, 25)
    q3 = np.percentile(final_scores, 75)
    for i in range(len(final_scores)):
        if q1 <= final_scores[i] <= q3:
            iq_indices.append(i)
    return iq_indices


def mean(curves):
    y_axis_values = [curve.values for curve in curves]
    mean_value = np.mean(np.stack(y_axis_values), axis=0)
    return mean_value

def compute_se(curves):
    num_datapoints = float(len(curves))
    y_axis_values = [curve.values for curve in curves]
    sample_std = np.std(np.stack(y_axis_values), axis=0, ddof=1)
    standard_error = sample_std / np.sqrt(num_datapoints)
    return standard_error
   

def compute_confidence_increment(curves, confidence_level=0.95):
    se = compute_se(curves)
    z_score = compute_z_score(confidence_level)
    confidence_increment = z_score * se
    return confidence_increment
