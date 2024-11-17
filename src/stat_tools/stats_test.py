"""
Module for performing statistical tests on datasets.

This module provides the StatisticalTest class for preparing data and performing various statistical tests,
including normality tests, variance tests, and appropriate statistical tests based on data characteristics.
It also includes methods for performing post-hoc tests when necessary.
"""

import numpy as np
from scipy import stats
from datasets import Dataset, DatasetDict
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from typing import Union, List, Tuple, Dict, Any


class StatisticalTest:
    """
    A class for performing statistical tests on datasets.

    This class provides methods to prepare data, perform normality and variance tests,
    select and perform appropriate statistical tests, and execute post-hoc tests if necessary.

    Attributes:
        data (Dataset): The dataset on which statistical tests will be performed.
    """

    def __init__(
            self, 
            data: Union[DatasetDict, Dataset]
            ):
        """
        Initialize the StatisticalTest with the provided dataset.

        Parameters:
            data (Union[DatasetDict, Dataset]): The dataset on which to perform statistical tests.
                If a DatasetDict is provided, it combines all splits into a single Dataset.
        """
        if isinstance(data, DatasetDict):
            combined_data = {}
            for split in data:
                for key in data[split].features.keys():
                    combined_data.setdefault(key, []).extend(data[split][key])
            self.data = Dataset.from_dict(combined_data)
        else:
            self.data = data

    def _prepare_data(
            self, 
            category_col: str, 
            wer_col: str
            ) -> Tuple[List[Any], List[float], List[Any], List[np.ndarray]]:
        """
        Prepare data by grouping WER values according to categories.

        Parameters:
            category_col (str): The name of the column containing category labels.
            wer_col (str): The name of the column containing WER values.

        Returns:
            Tuple containing:
                categories (List[Any]): List of category labels.
                wer_values (List[float]): List of WER values.
                unique_categories (List[Any]): Sorted list of unique category labels.
                wer_groups (List[np.ndarray]): List of WER values grouped by category.
        """
        categories = self.data[category_col]
        wer_values = self.data[wer_col]
        unique_categories = sorted(set(categories))
        wer_groups = [
            np.array([wer for cat, wer in zip(categories, wer_values) if cat == category])
            for category in unique_categories
        ]
        return categories, wer_values, unique_categories, wer_groups

    def _perform_normality_tests(
        self, 
        wer_groups: List[np.ndarray], 
        alpha: float
    ) -> Tuple[List[Dict[str, float]], List[bool]]:
        """
        Perform Shapiro-Wilk normality tests on each group of WER values.

        Parameters:
            wer_groups (List[np.ndarray]): List of WER values grouped by category.
            alpha (float): Significance level for the normality tests.

        Returns:
            Tuple containing:
                shapiro_results (List[Dict[str, float]]): List of dictionaries with test statistic and p-value for each group.
                normal_distributions (List[bool]): List indicating whether each group follows a normal distribution.
        """
        shapiro_results = []
        normal_distributions = []
        for group in wer_groups:
            if len(group) < 3:
                stat, p_value = np.nan, np.nan
                is_normal = False
            else:
                stat, p_value = stats.shapiro(group)
                is_normal = p_value > alpha
            shapiro_results.append({'statistic': stat, 'p_value': p_value})
            normal_distributions.append(is_normal)
        return shapiro_results, normal_distributions

    def _perform_variance_test(
            self, 
            wer_groups: List[np.ndarray]
            ) -> Dict[str, float]:
        """
        Perform Levene's test for equal variances across groups.

        Parameters:
            wer_groups (List[np.ndarray]): List of WER values grouped by category.

        Returns:
            Dict[str, float]: Dictionary containing the test statistic and p-value.
        """
        stat, p_value = stats.levene(*wer_groups, center='mean')
        return {'statistic': stat, 'pvalue': p_value}

    def _select_and_perform_test(
        self,
        wer_groups: List[np.ndarray],
        normal_distributions: List[bool],
        variance_p_value: float,
        alpha: float
    ) -> Tuple[str, float, float]:
        """
        Select and perform the appropriate statistical test based on data characteristics.

        Parameters:
            wer_groups (List[np.ndarray]): List of WER values grouped by category.
            normal_distributions (List[bool]): List indicating whether each group follows a normal distribution.
            variance_p_value (float): P-value from Levene's test for equal variances.
            alpha (float): Significance level for the tests.

        Returns:
            Tuple containing:
                test_used (str): Name of the statistical test performed.
                test_statistic (float): Test statistic value.
                p_value (float): P-value from the statistical test.
        """
        num_categories = len(wer_groups)
        if num_categories == 2:
            if all(normal_distributions):
                if variance_p_value > alpha:
                    test_statistic, p_value = stats.ttest_ind(wer_groups[0], wer_groups[1], equal_var=True)
                    test_used = 'Independent Two-Sample t-test'
                else:
                    test_statistic, p_value = stats.ttest_ind(wer_groups[0], wer_groups[1], equal_var=False)
                    test_used = "Welch's t-test"
            else:
                test_statistic, p_value = stats.mannwhitneyu(wer_groups[0], wer_groups[1], alternative='two-sided')
                test_used = 'Mann-Whitney U test'
        else:
            if all(normal_distributions):
                if variance_p_value > alpha:
                    test_statistic, p_value = stats.f_oneway(*wer_groups)
                    test_used = 'One-Way ANOVA'
                else:
                    test_statistic, p_value = stats.kruskal(*wer_groups)
                    test_used = 'Kruskal-Wallis H-test'
            else:
                test_statistic, p_value = stats.kruskal(*wer_groups)
                test_used = 'Kruskal-Wallis H-test'
        return test_used, test_statistic, p_value

    def _perform_post_hoc_tests(
        self,
        wer_groups: List[np.ndarray],
        unique_categories: List[Any],
        alpha: float
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform Tukey's Honest Significant Difference (HSD) post-hoc tests.

        Parameters:
            wer_groups (List[np.ndarray]): List of WER values grouped by category.
            unique_categories (List[Any]): Sorted list of unique category labels.
            alpha (float): Significance level for the tests.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary containing the results of the Tukey HSD tests.
        """
        wer_values = np.concatenate(wer_groups)
        labels = np.concatenate([[cat]*len(group) for cat, group in zip(unique_categories, wer_groups)])
        tukey_result = pairwise_tukeyhsd(wer_values, labels, alpha=alpha)
        tukey_results_list = [
            {
                'Group1': tukey_result.groupsunique[pair[0]],
                'Group2': tukey_result.groupsunique[pair[1]],
                'Mean Diff': tukey_result.meandiffs[i],
                'p-value': tukey_result.pvalues[i],
                'Reject': tukey_result.reject[i]
            }
            for i, pair in enumerate(tukey_result._pairindices)
        ]
        return {'Tukey HSD Results': tukey_results_list}

    def perform_tests(
        self,
        category_col: str,
        wer_col: str,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform statistical tests on the dataset.

        Parameters:
            category_col (str): The name of the column containing category labels.
            wer_col (str): The name of the column containing WER values.
            alpha (float, optional): Significance level for the tests. Defaults to 0.05.

        Returns:
            Dict[str, Any]: Dictionary containing the results of the statistical tests.
        """
        categories, wer_values, unique_categories, wer_groups = self._prepare_data(category_col, wer_col)

        if not categories or not wer_values:
            raise ValueError("No data available for statistical tests.")
        
        if any(w is None or (isinstance(w, float) and np.isnan(w)) for w in wer_values):
            raise ValueError("WER values contain missing entries.")
        
        if any(c is None for c in categories):
            raise ValueError("Category labels contain missing entries.")

        if len(unique_categories) < 2:
            raise ValueError("At least two categories are required for statistical tests.")
        try:
            wer_values = [float(w) for w in wer_values]
        except ValueError:
            raise ValueError("WER values must be numeric.")
        _, normal_distributions = self._perform_normality_tests(wer_groups, alpha)
        variance_test = self._perform_variance_test(wer_groups)
        test_used, test_statistic, p_value = self._select_and_perform_test(
            wer_groups, normal_distributions, variance_test['pvalue'], alpha
        )
        results = {
            'Test Used': test_used,
            'Test Statistic': test_statistic,
            'p-value': p_value,
            'Number of Categories': len(unique_categories),
            'Significant': p_value < alpha,
        }
        if results['Significant'] and len(unique_categories) > 2:
            post_hoc_results = self._perform_post_hoc_tests(wer_groups, unique_categories, alpha)
            results['Post-hoc Tests'] = post_hoc_results
        return results
