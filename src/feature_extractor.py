"""
Feature extraction from knowledge graphs for machine learning.
Extracts graph-based features to enhance XGBoost model performance.
"""

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, stddev, skewness, kurtosis
from graphframes import GraphFrame
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class GraphFeatureExtractor:
    """Extracts features from knowledge graphs for ML models."""
    
    def __init__(self, spark_session: SparkSession):
        self.spark = spark_session
        self.graph_features = {}
        
    def extract_node_features(self, graph: GraphFrame) -> pd.DataFrame:
        """Extract node-level features from the graph."""
        
        # Get basic degree information
        degrees_info = graph.degrees
        in_degrees_info = graph.inDegrees
        out_degrees_info = graph.outDegrees
        
        # Run PageRank
        pagerank_result = graph.pageRank(resetProbability=0.15, maxIter=10)
        pagerank_vertices = pagerank_result.vertices.select("id", "pagerank")
        
        # Join all node features
        node_features = degrees_info.join(in_degrees_info, "id", "left_outer") \
                                  .join(out_degrees_info, "id", "left_outer") \
                                  .join(pagerank_vertices, "id", "left_outer")
        
        # Fill null values with 0
        node_features = node_features.fillna(0)
        
        # Convert to pandas for easier manipulation
        node_features_pd = node_features.toPandas()
        
        # Calculate additional features
        if len(node_features_pd) > 0:
            node_features_pd['degree_centrality'] = node_features_pd['degree'] / (len(node_features_pd) - 1)
            node_features_pd['in_out_ratio'] = np.where(
                node_features_pd['outDegree'] > 0,
                node_features_pd['inDegree'] / node_features_pd['outDegree'],
                node_features_pd['inDegree']
            )
        
        logger.info(f"Extracted node features for {len(node_features_pd)} nodes")
        return node_features_pd
    
    def extract_graph_level_features(self, graph: GraphFrame) -> Dict[str, float]:
        """Extract graph-level features."""
        
        num_vertices = graph.vertices.count()
        num_edges = graph.edges.count()
        
        # Basic graph metrics
        density = (2 * num_edges) / (num_vertices * (num_vertices - 1)) if num_vertices > 1 else 0
        avg_degree = (2 * num_edges) / num_vertices if num_vertices > 0 else 0
        
        # Connected components
        try:
            components = graph.connectedComponents()
            num_components = components.select("component").distinct().count()
        except Exception as e:
            logger.warning(f"Could not compute connected components: {e}")
            num_components = 1
        
        # Triangle count (if graph is small enough)
        num_triangles = 0
        if num_vertices < 1000:  # Avoid expensive computation on large graphs
            try:
                triangles = graph.triangleCount()
                num_triangles = triangles.agg({"triangleCount": "sum"}).collect()[0][0] or 0
            except Exception as e:
                logger.warning(f"Could not compute triangle count: {e}")
        
        # Clustering coefficient approximation
        clustering_coeff = (3 * num_triangles) / (num_edges * (avg_degree - 1)) if avg_degree > 1 else 0
        
        graph_features = {
            'num_vertices': float(num_vertices),
            'num_edges': float(num_edges),
            'density': density,
            'avg_degree': avg_degree,
            'num_components': float(num_components),
            'num_triangles': float(num_triangles),
            'clustering_coefficient': clustering_coeff,
            'connectivity': 1.0 - (num_components / num_vertices) if num_vertices > 0 else 0
        }
        
        logger.info(f"Extracted {len(graph_features)} graph-level features")
        return graph_features
    
    def extract_statistical_features(self, df) -> Dict[str, Dict[str, float]]:
        """Extract statistical features from the original dataset."""
        
        statistical_features = {}
        numeric_columns = [col_name for col_name in df.columns if col_name != 'target']
        
        for col_name in numeric_columns:
            try:
                # Basic statistics
                stats = df.select(col_name).describe().collect()
                mean_val = float([row for row in stats if row[0] == 'mean'][0][1])
                std_val = float([row for row in stats if row[0] == 'stddev'][0][1])
                min_val = float([row for row in stats if row[0] == 'min'][0][1])
                max_val = float([row for row in stats if row[0] == 'max'][0][1])
                
                # Advanced statistics
                skew_val = df.select(skewness(col(col_name))).collect()[0][0] or 0
                kurt_val = df.select(kurtosis(col(col_name))).collect()[0][0] or 0
                
                statistical_features[col_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'range': max_val - min_val,
                    'skewness': float(skew_val),
                    'kurtosis': float(kurt_val),
                    'cv': std_val / abs(mean_val) if mean_val != 0 else 0  # Coefficient of variation
                }
                
            except Exception as e:
                logger.warning(f"Could not compute statistics for {col_name}: {e}")
                statistical_features[col_name] = {
                    'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'range': 0,
                    'skewness': 0, 'kurtosis': 0, 'cv': 0
                }
        
        logger.info(f"Extracted statistical features for {len(statistical_features)} columns")
        return statistical_features
    
    def create_feature_matrix(self, original_df, graph: GraphFrame, 
                            include_graph_features: bool = True) -> pd.DataFrame:
        """Create a comprehensive feature matrix combining original and graph features."""
        
        # Convert original data to pandas
        original_features = original_df.toPandas()
        
        feature_matrix = original_features.copy()
        
        if include_graph_features:
            # Extract graph-level features
            graph_features = self.extract_graph_level_features(graph)
            
            # Add graph features as columns (same value for all rows)
            for feature_name, feature_value in graph_features.items():
                feature_matrix[f'graph_{feature_name}'] = feature_value
            
            # Extract statistical features
            stat_features = self.extract_statistical_features(original_df)
            
            # Add aggregated statistical features
            for col_name, stats in stat_features.items():
                for stat_name, stat_value in stats.items():
                    feature_matrix[f'{col_name}_{stat_name}'] = stat_value
        
        logger.info(f"Created feature matrix with shape {feature_matrix.shape}")
        return feature_matrix
    
    def extract_correlation_features(self, correlations: List[tuple]) -> Dict[str, float]:
        """Extract features from correlation analysis."""
        
        if not correlations:
            return {'avg_correlation': 0, 'max_correlation': 0, 'num_strong_correlations': 0}
        
        corr_values = [abs(corr) for _, _, corr in correlations]
        
        correlation_features = {
            'avg_correlation': np.mean(corr_values),
            'max_correlation': np.max(corr_values),
            'min_correlation': np.min(corr_values),
            'std_correlation': np.std(corr_values),
            'num_strong_correlations': sum(1 for corr in corr_values if corr > 0.7),
            'num_moderate_correlations': sum(1 for corr in corr_values if 0.3 < corr <= 0.7),
            'correlation_density': len(correlations) / (16 * 15 / 2)  # For 16 features
        }
        
        return correlation_features
    
    def get_feature_importance_from_graph(self, graph: GraphFrame, 
                                        method: str = 'pagerank') -> pd.DataFrame:
        """Get feature importance based on graph structure."""
        
        if method == 'pagerank':
            pagerank_result = graph.pageRank(resetProbability=0.15, maxIter=10)
            importance_df = pagerank_result.vertices.select("id", "pagerank").toPandas()
            importance_df.columns = ['feature', 'importance']
            
        elif method == 'degree':
            degrees = graph.degrees.toPandas()
            degrees.columns = ['feature', 'importance']
            importance_df = degrees
            
        else:
            raise ValueError(f"Unknown importance method: {method}")
        
        # Normalize importance scores
        if len(importance_df) > 0:
            max_importance = importance_df['importance'].max()
            if max_importance > 0:
                importance_df['importance'] = importance_df['importance'] / max_importance
        
        return importance_df.sort_values('importance', ascending=False)
