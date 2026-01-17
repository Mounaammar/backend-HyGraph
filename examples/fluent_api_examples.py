"""
HyGraph - Fluent API Examples

This script demonstrates the core operators available in HyGraph:
1. CRUD Operations (Create, Read, Update, Delete)
2. Querying with Filters
3. Temporal Snapshots
4. Snapshot Sequences & TSGen
5. Pattern Matching
6. SubHyGraph

Prerequisites:
    1. Start the database: cd docker && docker-compose up -d
    2. Install dependencies: pip install -r requirements.txt
"""

from datetime import datetime, timedelta
from pathlib import Path

# =============================================================================
# SETUP
# =============================================================================

from hygraph_core.hygraph import HyGraph
from hygraph_core.model.timeseries import TimeSeries, TimeSeriesMetadata

# Connect to HyGraph
hg = HyGraph("postgresql://postgres:postgres@127.0.0.1:5433/hygraph")
print(f"Connected to HyGraph: {hg.stats()}")


# =============================================================================
# 1. CRUD OPERATIONS
# =============================================================================
print("\n" + "="*60)
print("1. CRUD OPERATIONS")
print("="*60)

# --- CREATE NODE ---
# Simple node with static properties
hg.create_node('station_A') \
    .with_label('Station') \
    .with_property('capacity', 50) \
    .with_property('name', 'Central Station') \
    .with_property('lat', 40.7128) \
    .with_property('lon', -74.0060) \
    .valid_from('2024-01-01') \
    .valid_until('2100-12-31') \
    .create()

print("✓ Created node: station_A")

# Node with time series property
timestamps = [datetime.now() - timedelta(hours=i) for i in range(24)]
data = [[float(20 + (i % 15))] for i in range(24)]

ts_bikes = TimeSeries(
    tsid='ts_stationB_bikes',
    timestamps=timestamps,
    variables=['num_bikes'],
    data=data,
    metadata=TimeSeriesMetadata(owner_id='station_B', element_type='node')
)

hg.create_node('station_B') \
    .with_label('Station') \
    .with_property('capacity', 30) \
    .with_ts_property('num_bikes_available', ts_bikes) \
    .create()

print("✓ Created node with time series: station_B")

# --- CREATE EDGE ---
hg.create_edge('station_A', 'station_B', 'trip_001') \
    .with_label('Trip') \
    .with_property('distance', 2.5) \
    .with_property('duration', 15) \
    .valid_from('2024-06-01T10:00:00') \
    .valid_until('2024-06-01T10:15:00') \
    .create()

print("✓ Created edge: trip_001")

# --- UPDATE ---
hg.update_node('station_A') \
    .set_property('capacity', 60) \
    .execute()

print("✓ Updated station_A capacity to 60")

# --- DELETE (soft) ---
# hg.delete_node('station_A').execute()  # Soft delete
# hg.delete_node('station_A').hard().execute()  # Hard delete


# =============================================================================
# 2. QUERYING
# =============================================================================
print("\n" + "="*60)
print("2. QUERYING")
print("="*60)

# --- Basic Query ---
stations = hg.query().nodes(label='Station').execute()
print(f"✓ Found {len(stations)} stations")

# --- Get Specific Node ---
node = hg.query().node('station_A').execute()
if node:
    print(f"✓ Node station_A: capacity={node.get_static_property('capacity')}")

# --- Filter by Static Property ---
high_capacity = hg.query() \
    .nodes(label='Station') \
    .where(lambda n: n.get_static_property('capacity', 0) > 40) \
    .execute()
print(f"✓ High capacity stations (>40): {len(high_capacity)}")

# --- Filter by Time Series Property ---
# Using lambda with TimeSeries methods
low_availability = hg.query() \
    .nodes(label='Station') \
    .where_ts('num_bikes_available', lambda ts: ts.mean() < 25) \
    .execute()
print(f"✓ Stations with low avg bikes (<25): {len(low_availability)}")

# Using where_ts_agg
result = hg.query() \
    .nodes(label='Station') \
    .where_ts_agg('num_bikes_available', 'mean', '<', 25) \
    .execute()
print(f"✓ Same query with where_ts_agg: {len(result)}")

# --- Get Time Series Property ---
ts = hg.query().node('station_B').ts_property('num_bikes_available')
if ts:
    print(f"✓ Time series for station_B: mean={ts.mean():.2f}, std={ts.std():.2f}")


# =============================================================================
# 3. TEMPORAL SNAPSHOTS
# =============================================================================
print("\n" + "="*60)
print("3. TEMPORAL SNAPSHOTS")
print("="*60)

# --- Point Snapshot ---
snap = hg.snapshot("2024-06-01T12:00:00", mode="hybrid")
print(f"✓ Point snapshot at 2024-06-01T12:00:00:")
print(f"   Nodes: {snap.count_nodes()}, Edges: {snap.count_edges()}")
print(f"   Density: {snap.density():.6f}")

# --- Interval Snapshot with Aggregation ---
snap_interval = hg.snapshot(
    when=("2024-06-01T00:00:00", "2024-06-02T00:00:00"),
    mode="hybrid",
    ts_handling="aggregate",
    aggregation_fn="mean"
)
print(f"✓ Interval snapshot (1 day):")
print(f"   Nodes: {snap_interval.count_nodes()}, Edges: {snap_interval.count_edges()}")

# --- Snapshot Metrics ---
cc = snap.connected_components()
print(f"✓ Connected components: {cc.count}, Biggest: {cc.biggest}")


# =============================================================================
# 4. SNAPSHOT SEQUENCES & TSGEN
# =============================================================================
print("\n" + "="*60)
print("4. SNAPSHOT SEQUENCES & TSGEN")
print("="*60)

# --- Create Snapshot Sequence ---
snapshots = hg.snapshot_sequence(
    start="2024-05-20",
    end="2024-05-25",
    every=timedelta(days=1),
    mode="hybrid"
)
print(f"✓ Created {len(snapshots)} snapshots")

#  Generate Time Series from Graph Evolution
ts = snapshots.tsgen()

# Global node metrics
node_count = ts.global_.nodes.count(label="Station")
print(f"✓ Node count time series: {len(node_count.timestamps)} points")

# Average degree over time
avg_degree = ts.global_.nodes.degree(label="Station").avg()
print(f"✓ Average degree over time: {len(avg_degree.timestamps)} points")

# Property aggregation over time
avg_capacity = ts.global_.nodes.property("capacity").avg()
print(f"✓ Average capacity over time")

# Graph-level metrics
density_ts = ts.global_.graph.density()
print(f"✓ Density time series: {len(density_ts.timestamps)} points")

# Edge count between specific nodes
trip_count = ts.global_.edges.between("station_A", "station_B").count()
print(f"✓ Trip count between A→B: {len(trip_count.timestamps)} points")

# Per-entity degree evolution
degree_ts = ts.entities.nodes.degree(
    node_id="station_A",
    label="Station",
    direction="both"
)
print(f"✓ Degree evolution for station_A")


# =============================================================================
# 5. PATTERN MATCHING
# =============================================================================
print("\n" + "="*60)
print("5. PATTERN MATCHING")
print("="*60)

# --- Basic Pattern: (n1)-[e]->(n2) ---
matches = hg.query().match_pattern() \
    .node("n1", label="Station") \
    .edge("e", label="Trip") \
    .node("n2", label="Station") \
    .limit(10) \
    .execute()
print(f"✓ Found {len(matches)} (Station)-[Trip]->(Station) patterns")

# --- Pattern with Static Constraints ---
matches = hg.query().match_pattern() \
    .node("n1", label="Station") \
    .edge("e", label="Trip") \
    .node("n2", label="Station") \
    .where_node("n1", static={"capacity": (">", 40)}) \
    .limit(10) \
    .execute()
print(f"✓ Patterns where source capacity > 40: {len(matches)}")

# --- Pattern with Cross-Entity Constraint ---
matches = hg.query().match_pattern() \
    .node("n1", label="Station") \
    .edge("e", label="Trip") \
    .node("n2", label="Station") \
    .where_cross("n2.capacity", "<", "n1.capacity") \
    .limit(10) \
    .execute()
print(f"✓ Patterns where n2.capacity < n1.capacity: {len(matches)}")

# --- Pattern with Time Series Constraints ---
matches = hg.query().match_pattern() \
    .node("n1", label="Station") \
    .edge("e", label="Trip") \
    .node("n2", label="Station") \
    .where_node("n1", ts={"num_bikes_available": ("mean", ">", 10)}) \
    .where_node("n2", ts={"num_bikes_available": ("mean", "<", 20)}) \
    .limit(10) \
    .execute()
print(f"✓ Patterns with TS constraints: {len(matches)}")

# --- Pattern with Time Series Similarity ---
matches = hg.query().match_pattern() \
    .node("n1", label="Station") \
    .edge("e") \
    .node("n2", label="Station") \
    .where_node("n1", ts_similar={"ts_name": "num_bikes_available", "template": "increasing"}) \
    .limit(5) \
    .execute()
print(f"✓ Patterns with increasing TS trend: {len(matches)}")

# --- Cross Time Series Constraints ---
correlated = hg.query().match_pattern() \
    .node("n1", label="Station") \
    .edge("e", label="Trip") \
    .node("n2", label="Station") \
    .where_cross_ts("n1.num_bikes_available", "correlates", 
                    "n2.num_bikes_available", threshold=0.5) \
    .limit(5) \
    .execute()
print(f"✓ Correlated station pairs: {len(correlated)}")


# =============================================================================
# 6. SUBHYGRAPH
# =============================================================================
print("\n" + "="*60)
print("6. SUBHYGRAPH")
print("="*60)

# --- Create SubHyGraph from Query ---
high_cap_stations = hg.query() \
    .nodes("Station") \
    .where(lambda n: n.get_static_property('capacity', 0) > 40) \
    .execute()

if high_cap_stations:
    node_ids = {st.oid for st in high_cap_stations}
    
    sub_hg = hg.subHygraph(
        node_ids=node_ids,
        name="high_capacity_stations",
        filter_query="capacity > 40"
    )
    
    print(f"✓ Created subgraph: {sub_hg}")
    print(f"   Is subgraph: {sub_hg.properties.is_subgraph}")
    print(f"   Filter: {sub_hg.properties.filter_query}")
    
    # Operations work on subgraph
    sub_stats = sub_hg.stats()
    print(f"   Stats: {sub_stats}")


# =============================================================================
# 7. TIME SERIES PATTERN MATCHING
# =============================================================================
print("\n" + "="*60)
print("7. TIME SERIES PATTERN MATCHING")
print("="*60)

# --- Create Pattern Templates ---
spike = TimeSeries.create_pattern('spike', length=10)
drop = TimeSeries.create_pattern('drop', length=10)
print(f"✓ Created spike and drop patterns")

# --- Find Pattern in Time Series ---
ts = hg.query().node('station_B').ts_property('num_bikes_available')
if ts and hasattr(ts, 'find_pattern'):
    matches = ts.find_pattern(spike, top_k=3)
    print(f"✓ Found {len(matches)} spike patterns in station_B")
    for m in matches[:3]:
        print(f"   - At {m.start_time} (distance: {m.distance:.3f})")

# --- Filter Nodes by Pattern ---
stations_with_spikes = hg.query() \
    .nodes(label='Station') \
    .where_ts_pattern('num_bikes_available', 'spike', method='euclidean') \
    .limit(5) \
    .execute()
print(f"✓ Stations containing spike pattern: {len(stations_with_spikes)}")


# =============================================================================
# CLEANUP
# =============================================================================
print("\n" + "="*60)
print("CLEANUP")
print("="*60)

# Clean up test data (optional)
# hg.delete_node('station_A').hard().execute()
# hg.delete_node('station_B').hard().execute()

hg.close()
print("✓ Connection closed")
print("\nAll examples completed successfully!")
