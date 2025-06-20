"""
Knowledge graph builder using GraphFrames.
Creates nodes and edges from dataset relationships.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, monotonically_increasing_id
from graphframes import GraphFrame
import pandas as pd
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Builds knowledge graphs using GraphFrames from dataset relationships."""
    
    def __init__(self, spark_session: SparkSession):
        self.spark = spark_session
        self.graph = None
        self.vertices = None
        self.edges = None
        
    def create_feature_graph(self, correlations: List[Tuple[str, str, float]], 
                           feature_stats: pd.DataFrame) -> GraphFrame:
        """Create a graph where nodes are features and edges are correlations."""
        
        # Create vertices (nodes) - one for each feature
        feature_names = feature_stats.columns[1:].tolist()  # Skip 'summary' column
        vertices_data = []
        
        for i, feature in enumerate(feature_names):
            if feature in ['count', 'mean', 'stddev', 'min', 'max']:
                continue
                
            # Get statistics for this feature
            mean_val = float(feature_stats[feature_stats['summary'] == 'mean'][feature].iloc[0])
            std_val = float(feature_stats[feature_stats['summary'] == 'stddev'][feature].iloc[0])
            
            vertices_data.append({
                'id': feature,
                'name': feature,
                'type': 'feature',
                'mean': mean_val,
                'std': std_val,
                'importance': abs(mean_val) + std_val  # Simple importance metric
            })
        
        # Create vertices DataFrame
        self.vertices = self.spark.createDataFrame(vertices_data)
        
        # Create edges from correlations
        edges_data = []
        for src, dst, corr_val in correlations:
            edges_data.append({
                'src': src,
                'dst': dst,
                'relationship': 'correlation',
                'weight': abs(corr_val),
                'correlation': corr_val
            })
            
            # Add reverse edge for undirected graph
            edges_data.append({
                'src': dst,
                'dst': src,
                'relationship': 'correlation',
                'weight': abs(corr_val),
                'correlation': corr_val
            })
        
        # Create edges DataFrame
        self.edges = self.spark.createDataFrame(edges_data)
        
        # Create GraphFrame
        self.graph = GraphFrame(self.vertices, self.edges)
        
        logger.info(f"Created graph with {self.vertices.count()} vertices and {self.edges.count()} edges")
        return self.graph
    
    def create_data_instance_graph(self, df, sample_size: int = 100) -> GraphFrame:
        """Create a graph where nodes are data instances and edges are similarities."""
        
        # Sample data for performance
        sampled_df = df.sample(fraction=sample_size/df.count(), seed=42).limit(sample_size)
        
        # Add unique IDs
        sampled_df = sampled_df.withColumn("row_id", monotonically_increasing_id())
        
        # Create vertices (data instances)
        vertices_data = []
        pandas_sample = sampled_df.toPandas()
        
        for _, row in pandas_sample.iterrows():
            vertices_data.append({
                'id': f"instance_{int(row['row_id'])}",
                'type': 'data_instance',
                'target_value': float(row.get('target', 0)),
                'feature_sum': sum([float(row[col]) for col in row.index if col not in ['row_id', 'target']])
            })
        
        self.vertices = self.spark.createDataFrame(vertices_data)
        
        # Create edges based on similarity (simplified - using Euclidean distance)
        edges_data = []
        threshold = 2.0  # Similarity threshold
        
        for i in range(len(pandas_sample)):
            for j in range(i+1, len(pandas_sample)):
                row1 = pandas_sample.iloc[i]
                row2 = pandas_sample.iloc[j]
                
                # Calculate Euclidean distance
                features1 = [float(row1[col]) for col in row1.index if col not in ['row_id', 'target']]
                features2 = [float(row2[col]) for col in row2.index if col not in ['row_id', 'target']]
                
                distance = sum((f1 - f2) ** 2 for f1, f2 in zip(features1, features2)) ** 0.5
                similarity = 1 / (1 + distance)  # Convert distance to similarity
                
                if similarity > 0.5:  # Only create edges for similar instances
                    edges_data.append({
                        'src': f"instance_{int(row1['row_id'])}",
                        'dst': f"instance_{int(row2['row_id'])}",
                        'relationship': 'similarity',
                        'weight': similarity,
                        'distance': distance
                    })
        
        self.edges = self.spark.createDataFrame(edges_data)
        self.graph = GraphFrame(self.vertices, self.edges)
        
        logger.info(f"Created instance graph with {self.vertices.count()} vertices and {self.edges.count()} edges")
        return self.graph
    
    def run_pagerank(self, reset_probability: float = 0.15, max_iter: int = 20):
        """Run PageRank algorithm on the graph."""
        if self.graph is None:
            raise ValueError("No graph created. Call create_*_graph() first.")
        
        pagerank_result = self.graph.pageRank(resetProbability=reset_probability, maxIter=max_iter)
        return pagerank_result
    
    def find_connected_components(self):
        """Find connected components in the graph."""
        if self.graph is None:
            raise ValueError("No graph created. Call create_*_graph() first.")
        
        components = self.graph.connectedComponents()
        return components
    
    def count_triangles(self):
        """Count triangles in the graph."""
        if self.graph is None:
            raise ValueError("No graph created. Call create_*_graph() first.")
        
        triangles = self.graph.triangleCount()
        return triangles
    
    def get_graph_metrics(self) -> Dict[str, Any]:
        """Get basic graph metrics."""
        if self.graph is None:
            raise ValueError("No graph created. Call create_*_graph() first.")
        
        num_vertices = self.vertices.count()
        num_edges = self.edges.count()
        
        # Calculate density
        max_edges = num_vertices * (num_vertices - 1) / 2
        density = num_edges / max_edges if max_edges > 0 else 0
        
        return {
            'num_vertices': num_vertices,
            'num_edges': num_edges,
            'density': density,
            'avg_degree': (2 * num_edges) / num_vertices if num_vertices > 0 else 0
        }
    
    def get_node_degrees(self):
        """Calculate node degrees."""
        if self.graph is None:
            raise ValueError("No graph created. Call create_*_graph() first.")
        
        in_degrees = self.graph.inDegrees
        out_degrees = self.graph.outDegrees
        degrees = self.graph.degrees
        
        return {
            'in_degrees': in_degrees,
            'out_degrees': out_degrees,
            'degrees': degrees
        }
    
    def export_for_visualization(self, max_nodes: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Export graph data for visualization."""
        if self.graph is None:
            raise ValueError("No graph created. Call create_*_graph() first.")
        
        # Limit nodes for visualization performance
        vertices_sample = self.vertices.limit(max_nodes).toPandas()
        
        # Get edges for the sampled vertices
        vertex_ids = vertices_sample['id'].tolist()
        edges_filtered = self.edges.filter(
            (col('src').isin(vertex_ids)) & (col('dst').isin(vertex_ids))
        ).toPandas()
        
        return vertices_sample, edges_filtered
