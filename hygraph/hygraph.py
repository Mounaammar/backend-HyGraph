"""
HyGraph - Main Class

This is the class users interact with.
It's a thin coordination layer that provides:
1. Storage backends (AGE, TimescaleDB)
2. Fluent API for CRUD operations
3. Operators support via HybridStorage
4. Graph-level properties (static and temporal)
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Set, Union

from hygraph_core.ingest.csv_converter import json_to_csv
from hygraph_core.ingest.csv_loader import load_csv
from hygraph_core.model.graph_properties import HyGraphProperties
from hygraph_core.operators.hygraphDiff import HyGraphDiff
from hygraph_core.operators.temporal_snapshot import Snapshot
from hygraph_core.operators.snapshotSequence import SnapshotSequence
from hygraph_core.storage.filtered_stored import FilteredAGEStore, FilteredTimescaleStore
from hygraph_core.storage.sql import DBPool
from hygraph_core.storage.age import AGEStore
from hygraph_core.storage.timescale import TSStore
from hygraph_core.storage.hygraph_crud import HybridCRUD
from hygraph_core.storage.hybrid_storage import HybridStorage
from hygraph_core.model.timeseries import TimeSeries, TimeSeriesMetadata
from hygraph_core import SETTINGS
# Type alias for storage backends (original or filtered)
AGEStoreType = Union[AGEStore, FilteredAGEStore]
TSStoreType = Union[TSStore, FilteredTimescaleStore]

# Import fluent API builders (users never import these directly)
from hygraph_core.storage.fluent_api import (
    NodeCreator,
    EdgeCreator,
    QueryBuilder,
    EntityUpdater,
    EntityDeleter
)


class HyGraph:
    """
    Main HyGraph class.
    
    Architecture:
        User → HyGraph → [AGEStore, TSStore, HybridCRUD, HybridStorage] → PostgreSQL
    
    Properties:
        HyGraph supports both static and temporal properties at the graph level.
        
        # Static properties
        hg.properties.add_static_property("domain", "transportation")
        hg.properties.add_static_property("region", "NYC")
        
        # Temporal properties (computed metrics as time series)
        hg.properties.add_temporal_property("node_count", "ts_node_count_001")
    """
    
    # Type hints for instance attributes
    db: DBPool
    age: AGEStoreType
    timescale: TSStoreType
    crud: HybridCRUD
    hybrid: Optional[HybridStorage]
    graph_name: str
    properties: HyGraphProperties

    def __init__(self, connection_string: str, graph_name: str = "hygraph"):
        """
        Initialize HyGraph.

        Args:
            connection_string: PostgreSQL connection string
            graph_name: Apache AGE graph name
        """
        # Database pool
        self.db = DBPool(SETTINGS.dsn, SETTINGS.pool_min, SETTINGS.pool_max)

        # Storage backends
        self.age = AGEStore(self.db, graph=graph_name)
        self.timescale = TSStore(self.db)

        # Hybrid operations (combines age + timescale)
        self.crud = HybridCRUD(self.age, self.timescale)
        self.graph_name = graph_name
        
        # For operators (persistence ↔ memory)
        self.hybrid = HybridStorage(self.db, graph_name=graph_name)
        
        # Graph-level properties
        self.properties = HyGraphProperties(
            name=graph_name,
            is_subgraph=False
        )

    # =========================================================================
    # INGESTION Operations
    # =========================================================================

    def ingest_from_json(
            self,
            json_dir: Path,
            node_field_map: Dict[str, str],
            edge_field_map: Dict[str, str],
            output_dir: Optional[Path] = None,
            skip_age: bool = False,
            batch_size: int = 5000
    ) -> Dict[str, int]:
        """
        Ingest data from JSON files using YOUR ingestion package.

        This method:
        1. Converts JSON to CSV (using csv_converter.py)
        2. Loads CSV into database (using csv_loader.py)

        Args:
            json_dir: Directory containing JSON files
            node_field_map: Field mapping for nodes
            edge_field_map: Field mapping for edges
            output_dir: Optional outputs directory for CSV (default: temp dir)
            skip_age: If True, skip AGE loading (only time series)
            batch_size: Batch size for loading

        """
        import tempfile

        # Use temp directory if no outputs directory specified
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="hygraph_csv_"))

        # Step 1: Convert JSON to CSV (YOUR function)
        print(f"\n📝 Converting JSON to CSV...")
        json_to_csv(
            json_dir=json_dir,
            output_dir=output_dir,
            node_field_map=node_field_map,
            edge_field_map=edge_field_map
        )

        # Step 2: Load CSV into database (YOUR function)
        print(f"\n💾 Loading CSV into database...")
        stats = load_csv(
            csv_dir=output_dir,
            graph_name=self.graph_name,
            skip_age=skip_age,
            batch_size=batch_size
        )
        
        # Update properties
        self.properties.touch()

        return stats

    def ingest_from_csv(
            self,
            csv_dir: Path,
            skip_age: bool = False,
            batch_size: int = 5000
    ) -> Dict[str, int]:
        """
        Ingest data from CSV files.
        """
        print(f"\n  Loading CSV into database...")
        stats = load_csv(
            csv_dir=csv_dir,
            graph_name=self.graph_name,
            skip_age=skip_age,
            batch_size=batch_size
        )
        
        # Update properties
        self.properties.touch()

        return stats

    # =========================================================================
    # FLUENT API - CRUD Operations (Database)
    # =========================================================================

    def create_node(self, oid: Optional[str] = None) -> NodeCreator:
        """
        Create a new node (goes to DATABASE).

        Example:
            hg.create_node('s1')\\
                .with_label('Station')\\
                .with_property('capacity', 50)\\
                .with_ts_property('bikes', ts_data)\\
                .create()
        """
        return NodeCreator(self, oid)

    def create_edge(self, source: str, target: str, oid: Optional[str] = None) -> EdgeCreator:
        """
        Create a new edges (goes to DATABASE).

        Example:
            hg.create_edge('s1', 's2')\\
                .with_label('trip')\\
                .with_property('distance', 2.5)\\
                .create()
        """
        return EdgeCreator(self, source, target, oid)

    def create_timeseries(self, timeseires:TimeSeries):

        return self.timescale.add_timeseries(timeseires)


    def query(self) -> QueryBuilder:
        """
        Start a query (queries DATABASE).

        Example:
            results = hg.query()\\
                .nodes(label='Station')\\
                .where(lambda n: n['capacity'] > 50)\\
                .execute()
        """
        return QueryBuilder(self)

    def update_node(self, node_id: str) -> EntityUpdater:
        """
        Update a node (updates DATABASE).

        Example:
            hg.update_node('s1')\\
                .set_property('capacity', 60)\\
                .execute()
        """
        return EntityUpdater(self, 'node', node_id)

    def update_edge(self, edge_id: str) -> EntityUpdater:
        """
        Update an edges (updates DATABASE).
        """
        return EntityUpdater(self, 'edges', edge_id)

    def delete_node(self, node_id: str) -> EntityDeleter:
        """
        Delete a node (deletes from DATABASE).

        Example:
            # Soft delete
            hg.delete_node('s1').execute()

            # Hard delete
            hg.delete_node('s1').hard().execute()
        """
        return EntityDeleter(self, 'node', node_id)

    def delete_edge(self, edge_id: str) -> EntityDeleter:
        """Delete an edges (deletes from DATABASE)."""
        return EntityDeleter(self, 'edges', edge_id)

    # =========================================================================
    # ADVANCED QUERIES (Database) - THROUGH FLUENT API
    # =========================================================================
    def count_nodes(self, label: Optional[str] = None) -> int:
        """Count nodes (in DATABASE)."""
        return self.age.count_nodes(label)

    def count_edges(self, label: Optional[str] = None) -> int:
        """Count edges (in DATABASE)."""
        return self.age.count_edges(label)

    # =========================================================================
    # OPERATORS (System loads to memory automatically)
    # =========================================================================

    def shortest_path(self, source: str, target: str):
        """
        Find shortest path (system loads to memory automatically).

        Example:
            path = hg.shortest_path('s1', 's2')
        """
        # Load to memory if not already loaded
        if not self.hybrid._memory_loaded:
            self.hybrid.load_to_memory()

        # Use in-memory graph
        nx_graph, _ = self.hybrid.get_memory_backend()
        import networkx as nx
        return nx.shortest_path(nx_graph, source, target)

    def snapshot(
        self, 
        when, 
        mode: str = "hybrid",
        ts_handling: Optional[str] = None,
        aggregation_fn: str = "mean"
    ) -> Snapshot:
        """
        Create a snapshot of the graph at a specific time or interval.
        
        Args:
            when: Timestamp string OR tuple (start, end) for interval
            mode: "graph" or "hybrid"
            ts_handling: For intervals with hybrid mode - "aggregate" or "slice"
            aggregation_fn: Aggregation function for "aggregate" mode
        
        Returns:
            Snapshot object
        
        Example:
            # Point snapshot
            snap = hg.snapshot("2024-05-01T10:00:00", mode="hybrid")
            
            # Interval snapshot with aggregation
            snap = hg.snapshot(
                when=("2024-05-01", "2024-05-02"),
                mode="hybrid",
                ts_handling="aggregate",
                aggregation_fn="mean"
            )
        """
        return Snapshot(
            self,
            when, 
            mode,
            ts_handling=ts_handling,
            aggregation_fn=aggregation_fn
        )

    def snapshot_sequence(
        self,
        start: str,
        end: str,
        every: timedelta,
        mode: str = "hybrid"
    ) -> SnapshotSequence:
        """
        Create a sequence of snapshots for temporal analysis.
        
        This is the FACTORY METHOD that handles all the complexity of:
        - Generating timestamps based on interval
        - Creating snapshots at each timestamp
        - Assembling them into a SnapshotSequence
        
        Snapshots are discrete samples of graph state. TSGen computes
        metrics from these samples, producing discrete time series.
        
        Args:
            start: Start timestamp (ISO format)
            end: End timestamp (ISO format)
            every: Time interval between snapshots (timedelta)
            mode: "graph" or "hybrid" (default: "hybrid")
        
        Returns:
            SnapshotSequence ready for TSGen analysis
        
        Example:
            from datetime import timedelta
            
            snapshots = hg.snapshot_sequence(
                start="2024-05-01",
                end="2024-05-31",
                every=timedelta(days=1)
            )
            
            # Generate time series from snapshots
            ts = snapshots.tsgen()
            node_count = ts.global_.nodes.count(label="Station")
        """
        if not isinstance(every, timedelta):
            raise TypeError(f"every must be a timedelta, got: {type(every).__name__}")
        
        # Parse start/end times
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        
        # Generate snapshots
        snapshots = []
        current = start_dt
        
        while current < end_dt:
            snap = Snapshot(
                self,
                when=current.isoformat(),
                mode=mode
            )
            snapshots.append(snap)
            current += every
        
        # Create and return SnapshotSequence
        return SnapshotSequence(
            snapshots=snapshots,
            every=every,
            start_time=start,
            end_time=end
        )

    @staticmethod
    def hygraph_diff(snap1: Snapshot, snap2: Snapshot):

        return HyGraphDiff(snap1, snap2)

    def subHygraph(
        self, 
        node_ids: Set[any],
        name: str,
        filter_query: Optional[str] = None
    ) -> 'HyGraph':
        """
        Create a subHygraph containing only specified nodes.
        Edges are automatically included if both endpoints are in node_ids.

        Args:
            node_ids: Set of node UIDs to include
            name: Name for the subHygraph
            filter_query: Optional human-readable filter description

        Returns:
            New HyGraph instance with filtered storage

        Example:
            # Get high-capacity stations
            nodes = hg.query().nodes("Station").where(
                lambda n: n.get_static_property("capacity") > 40
            ).execute()

            node_ids = {n.uid for n in nodes}

            # Create subHygraph
            high_cap_graph = hg.subHygraph(
                node_ids, 
                name="high_capacity",
                filter_query="capacity > 40"
            )
            
            # Access properties
            high_cap_graph.properties.is_subgraph  # True
            high_cap_graph.properties.filter_node_ids  # {...}
        """
        # Create filtered storage layers
        filtered_age = FilteredAGEStore(self.age, node_ids)
        filtered_timescale = FilteredTimescaleStore(self.timescale, filtered_age)

        # Create new HyGraph with filtered stores
        sub_hg = HyGraph.__new__(HyGraph)
        sub_hg.age = filtered_age
        sub_hg.timescale = filtered_timescale
        sub_hg.graph_name = name
        sub_hg.db = self.db
        sub_hg.crud = HybridCRUD(filtered_age, filtered_timescale)
        sub_hg.hybrid = None  # Not supported for subgraphs yet
        
        # Create properties for subgraph
        sub_hg.properties = HyGraphProperties(
            name=name,
            is_subgraph=True,
            filter_node_ids=node_ids,
            filter_query=filter_query
        )

        return sub_hg

    # =========================================================================
    # MEMORY MANAGEMENT (Advanced users)
    # =========================================================================

    def load_to_memory(self):
        """
        Explicitly load data to memory (for advanced users).

        Example:
            hg.load_to_memory()
            nx_graph, ts_data = hg.get_memory_backend()
            # Work with NetworkX directly
        """
        self.hybrid.load_to_memory()

    def get_memory_backend(self):
        """
        Get in-memory graph and time series (for advanced users).

        Returns:
            (nx_graph, ts_data)
        """
        return self.hybrid.get_memory_backend()

    def persist_to_db(self):
        """
        Persist in-memory changes back to database (for advanced users).
        """
        self.hybrid.flush_to_database()

    # =========================================================================
    # UTILITY
    # =========================================================================

    def close(self):
        """Close database connections"""
        self.db.close()

    def stats(self):
        """Get statistics"""
        return {
            'nodes': self.age.count_nodes(),
            'edges': self.age.count_edges(),
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self):
        stats = self.stats()
        graph_type = "SubHyGraph" if self.properties.is_subgraph else "HyGraph"
        return f"{graph_type}({self.properties.name}, nodes={stats['nodes']}, edges={stats['edges']})"


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("HyGraph Quick Example")
    print("=" * 80)

    hg = HyGraph("postgresql://localhost:5432/hygraph")
    schema = {
        "Station": {
            "capacity": "int",
            "lat": "float",
            "lon": "float",
            "region_id": "int"
        },

    }
    # Define field mappings
    # These tell the converter which fields in your JSON correspond to node/edges properties

    node_field_map = {
        "oid": "station_id",
        "start_time": "start",
        "end_time": "end",
    }
    edge_field_map = {
        "source_id": "from",
        "target_id": "to",
        "start_time": "start",
        "end_time": "end",
    }
    json_to_csv(
        json_dir=Path("../inputFiles/json"),
        output_dir=Path("../inputFiles/csv3"),
        node_field_map=node_field_map,
        edge_field_map=edge_field_map
    )

    # Step 2: Ingest CSV into database
    stats = load_csv(
        csv_dir=Path("../inputFiles/csv3"),
        graph_name="hygraph",
        skip_age=False,
        schema=schema
    )
    print("Ingested:", stats)

    # Create nodes
    #hg.create_node('s1').with_label('Station').with_property('capacity', 50).create()
    #hg.create_node('s2').with_label('Station').with_property('capacity', 70).create()
    
    # Create edge with time series
    """ now = datetime.now()
    timestamps = [now - timedelta(hours=i) for i in range(24)]
    data = [[float(i * 2)] for i in range(24)]

    ts_bikes = TimeSeries(
        tsid='ts_e1_available_bike',
        timestamps=timestamps,
        variables=['available_bike'],
        data=data,
        metadata=TimeSeriesMetadata(owner_id='E1', element_type='edge', units='bikes')
    )
    ts_id=hg.create_timeseries(ts_bikes)
    edge =hg.query().edge("Trip_2212_2212_2024-05-17T12:36:00").execute()
    edge.add_temporal_property(hg=hg,name="available_bike", ts_id=ts_id)
    print(ts_id)
    measurements = hg.timescale.get_measurements(
        entity_uid=ts_id,
        variable="available_bike"
    )
    print(measurements)
    
    hg.create_edge('s1', 's2', 'E1').with_label('Trip').with_ts_property("available_bike", ts_bikes).create()
    #result = hg.query().nodes("Station").where(lambda n: n.get_static_property('ca') >= 60).execute()
    #print(f"\nStations with capacity >= 50: {len(result), result}")
    edge =hg.query().edge("Trip_2212_2212_2024-05-17T12:36:00").execute()
    print("Available temporal properties:", edge.temporal_properties)
    print("Available static properties:", edge.static_properties)


    print(edge.get_temporal_property('num_rides'))
    print("Edge OID:", edge.oid)
    print("Temporal properties (IDs):", edge.temporal_properties)
    print("Data loaded status:", edge.list_temporal_properties_status())

    # Check TimescaleDB directly
    measurements = hg.timescale.get_measurements(
        entity_uid=edge.oid,
        variable="num_rides"
    )
    print(f"Direct TimescaleDB query with entity_uid='{edge.oid}': {len(measurements) if measurements else 0} rows")

    # Also try with the ts_id
    ts_id = edge.get_temporal_property_id("num_rides")
    print(f"ts_id: {ts_id}")
    # Query

    
    # Snapshot

    sn1 = hg.snapshot(when="2024-06-01 00:00:00", mode='hybrid')
    sn2 = hg.snapshot(when="2024-06-06 00:00:00", mode='hybrid')
    print(f"\nSnapshot: { sn1.count_nodes()} nodes, { sn1.count_edges()} edges")
    print(f"Density: { sn1.density()}\n")
    print(f"Connected components : { sn1.connected_components()} ")
    print(f"number of nodes of sn2: {sn2.count_nodes()}")

    # Snapshot sequence + TSGen
    """"""snapshots = hg.snapshot_sequence(
        start="2024-06-00 00:00:00",
        end="2024-06-06 00:00:00",
        every=timedelta(hours=1)
    )""""""

    edge = hg.query().edge("Trip_2212_2212_2024-05-17T12:36:00").execute()
    print("Available temporal properties:", edge.temporal_properties)
    print("Available static properties:", edge.static_properties)
   
    measurements = hg.timescale.get_measurements(
        entity_uid=edge.oid,
        variable="num_rides"
    )
    print(measurements)
    print(edge.get_temporal_property('num_rides'))
    print("Edge OID:", edge.oid)
    print("Temporal properties (IDs):", edge.temporal_properties)
    print("Data loaded status:", edge.list_temporal_properties_status())


    snapshots = hg.snapshot_sequence(
        start="2024-05-18 00:00:00",
        end="2024-05-25 00:00:00",
        every=timedelta(days=1)
    )
    print(f"\nSnapshotSequence: {len(snapshots)} snapshots")

    ts = snapshots.tsgen()
    node_count_ev=ts.global_.nodes.count(label="Station")
    
    print("node with id 2: ",hg.query().nodes(label="Station").execute()[1])
    avg_degree=ts.entities.nodes.degree(node_id="2172",label="Station", weight="num_rides").max()
    print(f"Node count TimeSeries: {node_count_ev}")
    print(f"Average degree TimeSeries: {avg_degree}")"""
    high_cap_stations=hg.query().nodes("Station").where(lambda n: n.get_static_property('capacity') >60).execute()
    print('result',len(high_cap_stations))
    nodes_id= {st.oid for st in high_cap_stations}
    high_cap=hg.subHygraph(nodes_id,"high_capacity_stations")
    #high_cap.properties.create_temporal_property(hg,'test',node_count_ev,)
    print("subgraph name:",high_cap.properties.get_temporal_property("test"),"\n")
    print("subgraph: ",high_cap,"\n")
    spike = TimeSeries.create_pattern('spike', length=10)
    ts=hg.query().node("2212").ts_property('num_bikes_available')
    print(spike)
    print(ts)
    matches=ts.find_pattern(spike,top_k=5)
    for m in matches:
        print(f"Found at {m.start_time} (distance: {m.distance:.3f})")
    snapshots = high_cap.snapshot_sequence(
        start="2024-05-18 00:00:00",
        end="2024-05-25 00:00:00",
        every=timedelta(days=1)
    )
    print(f"\nSnapshotSequence: {len(snapshots)} snapshots")
    hg.close()
    print("\n✓ Done")
