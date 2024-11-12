from abc import ABC, abstractmethod
from typing import List, Dict, Any
import evaluate

class BiasModule(ABC):
    def __init__(
            self, 
            metadata_field: str, 
            metric: evaluate.Metric):
        """
        Initializes the BiasModule with the specified metadata field and metric.

        Args:
            metadata_field (str): The field in the dataset's metadata to evaluate bias on.
            metric (evaluate.Metric): The evaluate metric object to apply.
        """
        self.metadata_field = metadata_field
        self.metric = metric

    @abstractmethod
    def compute_bias(
        self, 
        predictions: List[str], 
        references: List[str], 
        metadata: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
        """
        Abstract method to compute bias based on the metadata.

        Args:
            predictions (List[str]): The predicted transcriptions.
            references (List[str]): The ground truth transcriptions.
            metadata (List[Dict[str, Any]]): The metadata associated with each example.

        Returns:
            Dict[str, Any]: A dictionary with bias statistics for the given metadata field.
        """
        pass

class CategoricalBiasModule(BiasModule):
    def __init__(
            self, 
            metadata_field: str, 
            metric: evaluate.Metric):
        super().__init__(metadata_field, metric)

    def compute_bias(
            self, 
            predictions: List[str], 
            references: List[str], 
            metadata: List[Dict[str, Any]]
            ) -> Dict[str, Any]:
        """
        Computes the metric for the overall population and each category based on the metadata field.

        Args:
            predictions (List[str]): The predicted transcriptions.
            references (List[str]): The ground truth transcriptions.
            metadata (List[Dict[str, Any]]): The metadata associated with each example.

        Returns:
            Dict[str, Any]: A dictionary containing the overall metric score and metric scores by category.
        """
        # Compute the metric for the overall population
        overall_result = self.metric.compute(predictions=predictions, references=references)

        # Group data by the specified metadata field
        category_results = {}
        category_data = {}

        for pred, ref, meta in zip(predictions, references, metadata):
            if self.metadata_field in meta:
                category = meta[self.metadata_field]

                if category not in category_data:
                    category_data[category] = {'predictions': [], 'references': []}
                
                category_data[category]['predictions'].append(pred)
                category_data[category]['references'].append(ref)

        # Compute the metric for each category
        for category, data in category_data.items():
            if data['predictions'] and data['references']:
                category_results[category] = self.metric.compute(
                    predictions=data['predictions'],
                    references=data['references']
                )

        return {
            "overall": overall_result,
            "by_category": category_results
        }
