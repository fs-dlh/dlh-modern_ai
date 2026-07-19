#!/usr/bin/env python3
""" Module to select the best ccp_alpha for pruning."""


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """ Select the best ccp_alpha and its corresponding classifier.

    Criteria in order:
    1. Highest test accuracy.
    2. Smallest absolute difference between train and test accuracy.
    3. Largest ccp_alpha (to promote simpler trees).

    Args:
        clfs : Trained DecisionTreeClassifier instances.
        train_scores : Training accuracies for each model.
        test_scores : Test accuracies for each model.
        ccp_alphas : ccp_alpha values corresponding to the models.

    Returns:
        tuple: (best_alpha, best_clf)
            best_alpha : The best ccp_alpha value.
            best_clf: The trained classifier associated with best_alpha.
    """
    max_test = max(test_scores)

    candidates = [i for i, j in enumerate(test_scores) if j == max_test]

    min_diff = float('inf')
    best = candidates[0]
    for i in candidates:
        diff = abs(train_scores[i] - test_scores[i])
        if diff < min_diff:
            min_diff = diff
            best = i
        elif diff == min_diff:
            if ccp_alphas[i] > ccp_alphas[best]:
                best = i

    return ccp_alphas[best], clfs[best]
