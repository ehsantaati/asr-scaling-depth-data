import numpy as np
import json
import scipy.stats as stats
import re


def median_absolute_deviation(data):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return mad


def robust_standard_deviation(data):
    mad = median_absolute_deviation(data)

    # Calculate the scale factor
    # 75th percentile of the standard normal distribution
    percentile_value = stats.norm.ppf(0.75)
    scale_factor = 1 / percentile_value

    robust_std = mad * scale_factor
    return robust_std


def load_jsons_from_file_list(file_path_list):
    outputs = []
    for p in file_path_list:
        with open(p, "r") as f:
            out = json.load(f)
        outputs.append(out)
    return outputs


def remove_after_pattern(text, pattern=r"model\.decoder\.layers\.\d+"):
    """Removes everything after the specified pattern in a string."""

    match = re.search(pattern, text)
    if match:
        return text[: match.end()]
    else:
        return text


def convert_mixed_list(mixed_list):
    """Converts a mixed list with layer patterns to a list with combined blocks and remaining elements.

    Args:
      mixed_list: The list to be converted.

    Returns:
      The converted list.
    """

    blocks = []
    remaining = []
    pattern = "decoder.layers."

    for item in mixed_list:
        if item.startswith(pattern):
            blocks.append(int(item[len(pattern) :]))
        else:
            remaining.append(item)
        

    if blocks:
        return remaining + [
            f"{pattern[:-1].replace('layers','blocks')}[{','.join(map(str, blocks))}]"
        ]
    else:
        return mixed_list