# File: hygraph_core/operators/__init__.py
"""
HyGraph Operators Module

Architecture:
- All operators have to_dict() for internal serialization
- All operators have to_api_response() for lazy API responses
- Snapshot/SnapshotSequence have to_api_response_full() for eager responses
- Static properties are ALWAYS separate from temporal properties

Exports:
- Snapshot, SnapshotNode, SnapshotEdge: Temporal snapshots of graph state
- SnapshotSequence: Ordered collection of snapshots
- TSGen: Time series generator from snapshots
- HyGraphDiff: Diff between two snapshots
- Graph metrics: density, connected_components, etc.
"""

from hygraph_core.operators.temporal_snapshot import Snapshot, SnapshotNode, SnapshotEdge
from hygraph_core.operators.snapshotSequence import SnapshotSequence
from hygraph_core.operators.TSGen import TSGen
from hygraph_core.operators.hygraphDiff import (
    HyGraphDiff, 
    HyGraphDiffNode, 
    HyGraphDiffEdge,
    QueryableDiffNode,
    QueryableDiffEdge,
    DiffQueryBuilder,
    PropertyChange,
    # Interval diff with time series comparison
    HyGraphIntervalDiff,
    TSComparison,
    QueryableIntervalDiffNode,
    IntervalDiffQueryBuilder,
    compute_ts_comparison
)

# Graph metrics (can be used directly or via Snapshot methods)
from hygraph_core.operators.graph_metrics import (
    density,
    connected_components,
    degree_distribution,
    ComponentResult
)

__all__ = [
    # Snapshot classes
    "Snapshot",
    "SnapshotNode", 
    "SnapshotEdge",
    "SnapshotSequence",
    
    # Operators
    "TSGen",
    
    # Point diff (HyGraphDiff)
    "HyGraphDiff",
    "HyGraphDiffNode",
    "HyGraphDiffEdge",
    "QueryableDiffNode",
    "QueryableDiffEdge",
    "DiffQueryBuilder",
    "PropertyChange",
    
    # Interval diff with time series comparison (HyGraphIntervalDiff)
    "HyGraphIntervalDiff",
    "TSComparison",
    "QueryableIntervalDiffNode",
    "IntervalDiffQueryBuilder",
    "compute_ts_comparison",
    
    # Graph metrics
    "density",
    "connected_components",
    "degree_distribution",
    "ComponentResult",
]
