"""
Data processing module for knowledge graph creation.
Handles loading, cleaning, and preprocessing of 16-column dataset.
"""

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, corr, when, isnan, isnull
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data loading and preprocessing for knowledge graph creation."""
    
    def __init__(self, spark_session: SparkSession):
        self.spark = spark_session
        self.df = None
        self.column_names = []
        
    def load_data(self, file_path: str, header: bool = True) -> None:
        """Load data from CSV file."""
        try:
            self.df = self.spark.read.csv(file_path, header=header, inferSchema=True)
            self.column_names = self.df.columns
            logger.info(f"Loaded data with {self.df.count()} rows and {len(self.column_names)} columns")
            logger.info(f"Columns: {self.column_names}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_sample_data(self, n_rows: int = 1000) -> None:
        """Create sample 16-column dataset for demonstration."""
        np.random.seed(42)
        
        # Generate correlated data to create meaningful relationships
        base_data = np.random.randn(n_rows, 4)
        
        # Create 16 columns with various relationships
        data = {
            'feature_1': base_data[:, 0],
            'feature_2': base_data[:, 0] * 0.8 + np.random.randn(n_rows) * 0.2,  # Correlated with feature_1
            'feature_3': base_data[:, 1],
            'feature_4': base_data[:, 1] * 0.7 + base_data[:, 0] * 0.3 + np.random.randn(n_rows) * 0.1,
            'feature_5': base_data[:, 2],
            'feature_6': base_data[:, 2] * 0.9 + np.random.randn(n_rows) * 0.1,
            'feature_7': base_data[:, 3],
            'feature_8': base_data[:, 3] * 0.6 + base_data[:, 2] * 0.4,
            'feature_9': np.random.randn(n_rows),  # Independent
            'feature_10': np.random.randn(n_rows),  # Independent
            'feature_11': base_data[:, 0] * base_data[:, 1],  # Interaction term
            'feature_12': np.sin(base_data[:, 0]) + np.random.randn(n_rows) * 0.1,
            'feature_13': np.exp(base_data[:, 2] * 0.1) + np.random.randn(n_rows) * 0.1,
            'feature_14': base_data[:, 3] ** 2 + np.random.randn(n_rows) * 0.1,
            'feature_15': (base_data[:, 0] + base_data[:, 1] + base_data[:, 2]) / 3,  # Average
            'target': (base_data[:, 0] * 0.3 + base_data[:, 1] * 0.2 + 
                      base_data[:, 2] * 0.25 + base_data[:, 3] * 0.25 + np.random.randn(n_rows) * 0.1)
        }
        
        # Convert to pandas DataFrame then to Spark DataFrame
        pandas_df = pd.DataFrame(data)
        self.df = self.spark.createDataFrame(pandas_df)
        self.column_names = self.df.columns
        logger.info(f"Created sample data with {n_rows} rows and {len(self.column_names)} columns")
    
    def clean_data(self) -> None:
        """Clean the dataset by handling missing values and outliers."""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        initial_count = self.df.count()
        
        # Remove rows with any null values
        self.df = self.df.dropna()
        
        # Remove outliers (values beyond 3 standard deviations)
        for col_name in self.column_names:
            if col_name != 'target':  # Don't remove outliers from target
                stats = self.df.select(col_name).describe().collect()
                mean_val = float([row for row in stats if row[0] == 'mean'][0][1])
                std_val = float([row for row in stats if row[0] == 'stddev'][0][1])
                
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                
                self.df = self.df.filter(
                    (col(col_name) >= lower_bound) & (col(col_name) <= upper_bound)
                )
        
        final_count = self.df.count()
        logger.info(f"Data cleaning: {initial_count} -> {final_count} rows ({initial_count - final_count} removed)")
    
    def compute_correlations(self, threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        """Compute pairwise correlations between features."""
        correlations = []
        numeric_cols = [col_name for col_name in self.column_names if col_name != 'target']
        
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                try:
                    corr_val = self.df.stat.corr(col1, col2)
                    if abs(corr_val) >= threshold:
                        correlations.append((col1, col2, corr_val))
                except Exception as e:
                    logger.warning(f"Could not compute correlation between {col1} and {col2}: {e}")
        
        logger.info(f"Found {len(correlations)} correlations above threshold {threshold}")
        return correlations
    
    def get_feature_statistics(self) -> pd.DataFrame:
        """Get statistical summary of all features."""
        stats_df = self.df.describe().toPandas()
        return stats_df
    
    def get_spark_dataframe(self):
        """Return the Spark DataFrame."""
        return self.df
    
    def get_pandas_dataframe(self) -> pd.DataFrame:
        """Convert Spark DataFrame to Pandas for compatibility."""
        return self.df.toPandas()
