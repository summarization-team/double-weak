import sys
import os
import unittest
import numpy as np
from datasets import Dataset, DatasetDict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from stat_tools.stats_test import StatisticalTest


class TestStatisticalTest(unittest.TestCase):
    """
    Test suite for the StatisticalTest class.

    This suite focuses on testing the input and output of the StatisticalTest class methods.
    """

    def setUp(self):
        """
        Set up test datasets for the unit tests.
        """
        # Create a simple Dataset
        self.dataset = Dataset.from_dict({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'wer': [0.1, 0.2, 0.15, 0.25, 0.2, 0.3]
        })

        # Create a DatasetDict with multiple splits
        self.dataset_dict = DatasetDict({
            'train': Dataset.from_dict({
                'category': ['A', 'A'],
                'wer': [0.1, 0.2]
            }),
            'test': Dataset.from_dict({
                'category': ['B', 'B'],
                'wer': [0.15, 0.25]
            }),
            'validation': Dataset.from_dict({
                'category': ['C', 'C'],
                'wer': [0.2, 0.3]
            })
        })

    def test_initialization_with_dataset(self):
        """
        Test initializing StatisticalTest with a Dataset object.
        """
        st = StatisticalTest(self.dataset)
        self.assertIsInstance(st.data, Dataset)
        self.assertEqual(len(st.data), 6)

    def test_initialization_with_dataset_dict(self):
        """
        Test initializing StatisticalTest with a DatasetDict object.
        """
        st = StatisticalTest(self.dataset_dict)
        self.assertIsInstance(st.data, Dataset)
        self.assertEqual(len(st.data), 6)

    def test_perform_tests_valid_input(self):
        """
        Test the perform_tests method with valid inputs.
        """
        st = StatisticalTest(self.dataset)
        results = st.perform_tests('category', 'wer')
        self.assertIn('Test Used', results)
        self.assertIn('p-value', results)
        self.assertIn('Significant', results)
        self.assertIn('Number of Categories', results)
        self.assertEqual(results['Number of Categories'], 3)

    def test_perform_tests_invalid_category_column(self):
        """
        Test the perform_tests method with an invalid category column name.
        """
        st = StatisticalTest(self.dataset)
        with self.assertRaises(KeyError):
            st.perform_tests('invalid_category', 'wer')

    def test_perform_tests_invalid_wer_column(self):
        """
        Test the perform_tests method with an invalid WER column name.
        """
        st = StatisticalTest(self.dataset)
        with self.assertRaises(KeyError):
            st.perform_tests('category', 'invalid_wer')

    def test_perform_tests_non_numeric_wer_values(self):
        """
        Test the perform_tests method with non-numeric WER values.
        """
        data_non_numeric = Dataset.from_dict({
            'category': ['A', 'A', 'B', 'B'],
            'wer': ['0.1', '0.2', 'invalid', '0.25']
        })
        st = StatisticalTest(data_non_numeric)
        with self.assertRaises(ValueError):
            st.perform_tests('category', 'wer')

    def test_perform_tests_single_category(self):
        """
        Test the perform_tests method with only one category present.
        """
        data_single_category = Dataset.from_dict({
            'category': ['A', 'A', 'A'],
            'wer': [0.1, 0.2, 0.15]
        })
        st = StatisticalTest(data_single_category)
        with self.assertRaises(ValueError):
            st.perform_tests('category', 'wer')

    def test_perform_tests_custom_alpha(self):
        """
        Test the perform_tests method with a custom alpha value.
        """
        st = StatisticalTest(self.dataset)
        results = st.perform_tests('category', 'wer', alpha=0.01)
        self.assertIn('Test Used', results)
        self.assertIn('p-value', results)
        self.assertFalse(results['Significant'])

    def test_perform_tests_empty_dataset(self):
        """
        Test the perform_tests method with an empty dataset.
        """
        empty_data = Dataset.from_dict({'category': [], 'wer': []})
        st = StatisticalTest(empty_data)
        with self.assertRaises(ValueError):
            st.perform_tests('category', 'wer')

    def test_perform_tests_missing_values(self):
        """
        Test the perform_tests method with missing values in the dataset.
        """
        data_with_nans = Dataset.from_dict({
            'category': ['A', 'B', 'C'],
            'wer': [0.1, np.nan, 0.3]
        })
        st = StatisticalTest(data_with_nans)
        with self.assertRaises(ValueError):
            st.perform_tests('category', 'wer')


if __name__ == '__main__':
    unittest.main()
