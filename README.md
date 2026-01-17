# HyGraph 
## Link to the demo video: https://cloud.scadsai.uni-leipzig.de/index.php/s/4sHYj96AFfDH6PT

**HyGraph** is a novel hybrid system that combines **temporal property graphs** with **time series data** using PostgreSQL with Apache AGE and TimescaleDB extensions. It enables hybrid queries and analysis.

## Key Features

- **Hybrid Data Model**: Nodes and edges can have both static properties and temporal properties (time series)
- **Temporal Snapshots**: Query graph state at any point in time or over intervals
- **Time Series Generation (TSGen)**: Compute new time series from graph evolution metrics
- **Hybrid Pattern Matching**: Find subgraph patterns with both structural and time series constraints
- **Fluent API**: Intuitive Python interface for all operations
- **SubHyGraph**: Create filtered subgraphs for focused analysis

##  Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Fluent API Examples](#fluent-api-examples)
  - [CRUD Operations](#crud-operations)
  - [Querying](#querying)
  - [Temporal Snapshots](#temporal-snapshots)
  - [Snapshot Sequences & TSGen](#snapshot-sequences--tsgen)
  - [Pattern Matching](#pattern-matching)
  - [SubHyGraph](#subhygraph)
- [Data Ingestion](#data-ingestion)
- [API Reference](#api-reference)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         HyGraph                             │
│                    (Main Python Class)                      │
├─────────────────────────────────────────────────────────────┤
│  Fluent API: create_node(), query(), snapshot(), etc.       │
├──────────────────────┬──────────────────────────────────────┤
│     Apache AGE       │           TimescaleDB                │
│  (Graph Structure)   │         (Time Series)                │
│  - Nodes & Edges     │  - Measurements table                │
│  - Cypher queries    │  - Hypertables                       │
│                      │  - Time-bucket aggregations          │
├──────────────────────┴──────────────────────────────────────┤
│                      PostgreSQL                             │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.11+
- Docker & Docker Compose

### 1. Start the Database

```bash
cd docker
docker-compose up -d
```

This starts PostgreSQL with both **Apache AGE** and **TimescaleDB** extensions pre-configured.

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Connection

Set environment variable or use default:

```bash
export HYGRAPH_DSN="postgresql://postgres:postgres@127.0.0.1:5433/hygraph"
```

Or use default: `postgresql://postgres:postgres@127.0.0.1:5433/hygraph`

## Quick Start

```python
from hygraph_core.hygraph import HyGraph
from datetime import timedelta

# Connect to HyGraph
hg = HyGraph()

# Check stats
print(hg.stats())  # {'nodes': 1850, 'edges': 45000}

# Query nodes
stations = hg.query().nodes(label="Station").execute()
print(f"Found {len(stations)} stations")

# Create a snapshot
snap = hg.snapshot("2024-06-01T12:00:00", mode="hybrid")
print(f"Snapshot: {snap.count_nodes()} nodes, {snap.count_edges()} edges")

# Close connection
hg.close()
```

## Fluent API Examples

### CRUD Operations

#### Create Nodes

```python
from datetime import datetime, timedelta
from hygraph_core.model import TimeSeries, TimeSeriesMetadata

# Simple node
hg.create_node('station_1') \
    .with_label('Station') \
    .with_property('capacity', 50) \
    .with_property('name', 'Central Station') \
    .valid_from('2024-01-01') \
    .valid_until('2100-12-31') \
    .create()

# Node with time series property
timestamps = [datetime.now() - timedelta(hours=i) for i in range(24)]
data = [[float(20 + i % 10)] for i in range(24)]

ts_bikes = TimeSeries(
    tsid='ts_station1_bikes',
    timestamps=timestamps,
    variables=['num_bikes'],
    data=data,
    metadata=TimeSeriesMetadata(owner_id='2210', element_type='node')
)

hg.create_node('2210') \
    .with_label('Station') \
    .with_property('capacity', 30) \
    .with_ts_property('num_bikes_available', ts_bikes) \
    .create()
```

#### Create Edges

```python
# Simple edge
hg.create_edge('2210', '2215', 'trip_001') \
    .with_label('Trip') \
    .with_property('distance', 2.5) \
    .with_property('duration', 15) \
    .valid_from('2024-06-01T10:00:00') \
    .valid_until('2024-06-01T10:15:00') \
    .create()

# Edge with time series
hg.create_edge('2210', '2215') \
    .with_label('Trip') \
    .with_ts_property('flow', flow_timeseries) \
    .create()
```

#### Update & Delete

```python
# Update node properties
hg.update_node('2210') \
    .set_property('capacity', 60) \
    .execute()

# Soft delete (sets end_time to now)
hg.delete_node('2210').execute()

# Hard delete (removes from database)
hg.delete_node('2210').hard().execute()
```

### Querying

#### Basic Queries

```python
# Get all stations
stations = hg.query().nodes(label='Station').execute()

# Get specific node
node = hg.query().node('2210').execute()

# Get node property
capacity = hg.query().node('2210').property('capacity')

# Get time series property
ts = hg.query().node('2210').ts_property('num_bikes_available')
print(ts)  # Beautiful table display
print(ts.mean(), ts.std(), ts.min(), ts.max())
```

#### Filtering with Static Properties

```python
# Filter by static property using lambda
high_capacity = hg.query() \
    .nodes(label='Station') \
    .where(lambda n: n.get_static_property('capacity') > 40) \
    .execute()

# Filter by multiple conditions
filtered = hg.query() \
    .nodes(label='Station') \
    .where(lambda n: n.get_static_property('capacity') > 30) \
    .where(lambda n: n.get_static_property('region_id') == 71) \
    .execute()
```

#### Filtering with Time Series Constraints

```python
# Using TimeSeries methods directly
low_availability = hg.query() \
    .nodes(label='Station') \
    .where_ts('num_bikes_available', lambda ts: ts.mean() < 10) \
    .execute()

# Complex time series conditions
volatile_stations = hg.query() \
    .nodes(label='Station') \
    .where_ts('num_bikes_available', lambda ts: ts.std() > 15) \
    .where_ts('num_bikes_available', lambda ts: ts.max() - ts.min() > 30) \
    .execute()


hg.query() \
    .nodes(label='Station') \
    .where_ts_agg('num_bikes_available', 'mean', '<', 10) \
    .execute()

# Time-bucketed aggregation
hg.query() \
    .nodes(label='Station') \
    .where_ts_bucketed(
        'num_bikes_available',
        interval='1 hour',
        agg_func='avg',
        condition='all',  # 'all', 'any', 'majority', 'avg'
        operator='<',
        value=10
    ) \
    .execute()
```

#### Pattern Matching in Time Series

```python
# Create a spike pattern
spike = TimeSeries.create_pattern('spike', length=10)

# Find stations containing the pattern
ts = hg.query().node('2210').ts_property('num_bikes_available')
matches = ts.find_pattern(spike, top_k=5)

for m in matches:
    print(f"Found at {m.start_time} (distance: {m.distance:.3f})")

# Filter nodes by pattern presence
stations_with_spikes = hg.query() \
    .nodes(label='Station') \
    .where_ts_pattern('num_bikes_available', 'spike', method='shape') \
    .execute()

# Template patterns: 'spike', 'drop', 'increasing', 'decreasing', 'stable'
```

### Temporal Snapshots

#### Point Snapshot

```python
# Graph mode: temporal_properties contain ts_ids (references)
snap_graph = hg.snapshot("2024-06-01T12:00:00", mode="graph")

# Hybrid mode: temporal_properties contain resolved scalar values
snap_hybrid = hg.snapshot("2024-06-01T12:00:00", mode="hybrid")

# Access snapshot data
nodes = snap_hybrid.get_all_nodes()
edges = snap_hybrid.get_all_edges()

# Metrics
print(f"Nodes: {snap_hybrid.count_nodes()}")
print(f"Edges: {snap_hybrid.count_edges()}")
print(f"Density: {snap_hybrid.density()}")
print(f"Components: {snap_hybrid.connected_components()}")
```

#### Interval Snapshot

```python
# Aggregate time series over interval
snap_agg = hg.snapshot(
    when=("2024-06-01T00:00:00", "2024-06-01T23:59:59"),
    mode="hybrid",
    ts_handling="aggregate",  # 'aggregate' or 'slice'
    aggregation_fn="mean"     # 'mean', 'sum', 'min', 'max'
)

# Slice: keep all time series points within interval
snap_slice = hg.snapshot(
    when=("2024-06-01", "2024-06-07"),
    mode="hybrid",
    ts_handling="slice"
)
```

### Snapshot Sequences & TSGen

Create sequences of snapshots and generate time series from graph evolution:

```python
from datetime import timedelta

# Create snapshot sequence (daily snapshots for a week)
snapshots = hg.snapshot_sequence(
    start="2024-05-01",
    end="2024-05-08",
    every=timedelta(days=1),
    mode="hybrid"
)

print(f"Created {len(snapshots)} snapshots")

# Generate time series from graph evolution
ts = snapshots.tsgen()

# Global metrics - aggregate across all entities
node_count = ts.global_.nodes.count(label="Station")
edge_count = ts.global_.edges.count(label="Trip")
avg_degree = ts.global_.nodes.degree(label="Station").avg()
max_degree = ts.global_.nodes.degree(label="Station").max()

# Property aggregations
avg_capacity = ts.global_.nodes.property("capacity").avg()
total_capacity = ts.global_.nodes.property("capacity").sum()

# Graph-level metrics
density = ts.global_.graph.density()
components = ts.global_.graph.connected_components()

# Edge aggregations between specific nodes
trip_count = ts.global_.edges.between("2210", "2215").count()
total_duration = ts.global_.edges.between("2210", "2215").property("duration").sum()

# Per-entity metrics
degree_ts = ts.entities.nodes.degree(
    node_id="2215",
    label="Station",
    direction="both"  # 'in', 'out', 'both'
)

# Weighted degree (using edge property as weight)
weighted_degree = ts.global_.nodes.degree(label="Station", weight="duration").avg()

# Access TimeSeries results
print(node_count)  # TimeSeries object
print(node_count.timestamps)
print(node_count.data)
```

### Pattern Matching

Find subgraph patterns with structural and time series constraints:

```python
# Find trips where destination has lower capacity than source
matches = hg.query().match_pattern() \
    .node("n1", label="Station") \
    .edge("e", label="Trip") \
    .node("n2", label="Station") \
    .where_node("n1", static={"capacity": (">", 40)}) \
    .where_node("n2", ts={"num_bikes_available": ("mean", "<", 10)}) \
    .where_cross("n2.capacity", "<", "n1.capacity") \
    .execute()

for match in matches:
    n1, e, n2 = match['n1'], match['e'], match['n2']
    print(f"{n1.oid} ({n1.get_static_property('capacity')}) -> "
          f"{n2.oid} ({n2.get_static_property('capacity')})")

# Cross time series constraints
correlated_pairs = hg.query().match_pattern() \
    .node("n1", label="Station") \
    .edge("e", label="Trip") \
    .node("n2", label="Station") \
    .where_cross_ts("n1.num_bikes_available", "correlates", 
                    "n2.num_bikes_available", threshold=0.7) \
    .execute()

# Time series similarity constraints
declining_pairs = hg.query().match_pattern() \
    .node("n1", label="Station") \
    .edge("e") \
    .node("n2", label="Station") \
    .where_node("n1", ts_similar={"ts_name": "num_bikes_available", "template": "decreasing"}) \
    .where_node("n2", ts_similar={"ts_name": "num_bikes_available", "template": "stable"}) \
    .execute()
```

### SubHyGraph

Create filtered subgraphs for focused analysis:

```python
# Filter high-capacity stations
high_cap_stations = hg.query() \
    .nodes("Station") \
    .where(lambda n: n.get_static_property('capacity') > 60) \
    .execute()

node_ids = {st.oid for st in high_cap_stations}

# Create subgraph
sub_hg = hg.subHygraph(
    node_ids=node_ids,
    name="high_capacity_stations",
    filter_query="capacity > 60"
)

print(sub_hg)  # SubHyGraph(high_capacity_stations, nodes=45, edges=120)
print(sub_hg.properties.is_subgraph)  # True

# All operations work on subgraph
sub_snapshots = sub_hg.snapshot_sequence(
    start="2024-05-01",
    end="2024-05-08",
    every=timedelta(days=1)
)
```

## Data Ingestion

### From JSON Files

```python
from pathlib import Path
from hygraph_core.ingest.csv_converter import json_to_csv
from hygraph_core.ingest.csv_loader import load_csv

# Define schema for node types
schema = {
    "Station": {
        "capacity": "int",
        "lat": "float",
        "lon": "float",
        "region_id": "int"
    }
}

# Field mappings
node_field_map = {
    "oid": "station_id",      # Which field is the unique ID
    "start_time": "start",    # Temporal validity start
    "end_time": "end"         # Temporal validity end
}

edge_field_map = {
    "source_id": "from",
    "target_id": "to",
    "start_time": "start",
    "end_time": "end"
}

# Step 1: Convert JSON to CSV
json_to_csv(
    json_dir=Path("./data/json"),
    output_dir=Path("./data/csv"),
    node_field_map=node_field_map,
    edge_field_map=edge_field_map
)

# Step 2: Load CSV into database
stats = load_csv(
    csv_dir=Path("./data/csv"),
    graph_name="hygraph",
    schema=schema,
    skip_age=False
)

print(f"Ingested: {stats}")
# {'nodes': 1850, 'edges': 45000, 'measurements': 2500000}
```

## API Reference

### HyGraph Class

| Method | Description |
|--------|-------------|
| `create_node(oid)` | Start fluent node creation |
| `create_edge(source, target, oid)` | Start fluent edge creation |
| `query()` | Start a query builder |
| `update_node(oid)` | Start fluent node update |
| `delete_node(oid)` | Start fluent node deletion |
| `snapshot(when, mode)` | Create temporal snapshot |
| `snapshot_sequence(start, end, every)` | Create snapshot sequence |
| `subHygraph(node_ids, name)` | Create filtered subgraph |
| `stats()` | Get node/edge counts |
| `close()` | Close database connection |

### Snapshot Class

| Method | Description |
|--------|-------------|
| `get_all_nodes()` | Get all nodes in snapshot |
| `get_all_edges()` | Get all edges in snapshot |
| `count_nodes()` | Count nodes |
| `count_edges()` | Count edges |
| `density()` | Compute graph density |
| `connected_components()` | Find connected components |
| `degree(node_id, direction)` | Compute node degree |

### TSGen Class

| Method | Description |
|--------|-------------|
| `global_.nodes.count(label)` | Node count over time |
| `global_.nodes.degree(label).avg()` | Average degree over time |
| `global_.nodes.property(name).avg()` | Property aggregation |
| `global_.edges.count(label)` | Edge count over time |
| `global_.edges.between(src, dst).count()` | Edges between nodes |
| `global_.graph.density()` | Graph density over time |
| `entities.nodes.degree(node_id)` | Per-node degree evolution |


