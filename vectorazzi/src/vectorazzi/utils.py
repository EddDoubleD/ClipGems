import torch


def normalize_vector(vector):
    """
    Takes a vector and returns its normalized version

    :param vector: original vector of arbitrary length
    :return: vector of the same direction, with length 1
    """
    return vector / torch.linalg.norm(vector)


