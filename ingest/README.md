# HyGraph Ingestion API

Two simple endpoints:
1. **POST /api/convert** - Convert JSON folder to CSV
2. **POST /api/ingest** - Load CSV folder into HyGraph

## Run the API

```bash
python -m hygraph_core.ingest.api
# Docs: http://localhost:8001/docs
```

---

## Endpoint 1: Convert JSON to CSV

```
POST /api/convert
```

**Request body:**
```json
{
  "input_dir": "../inputFiles/json",
  "output_dir": "../inputFiles/csv3",
  "node_field_map": {
    "oid": "station_id",
    "start_time": "start",
    "end_time": "end"
  },
  "edge_field_map": {
    "source_id": "from",
    "target_id": "to",
    "start_time": "start",
    "end_time": "end"
  }
}
```

**Response (success):**
```json
{
  "success": true,
  "input_dir": "../inputFiles/json",
  "output_dir": "../inputFiles/csv3",
  "stats": {
    "nodes": 100,
    "edges": 500,
    "measurements": 10000,
    "time_series": 400
  }
}
```

**Response (error):**
```json
{
  "success": false,
  "error": "Input directory not found"
}
```

---

## Endpoint 2: Ingest CSV to HyGraph

```
POST /api/ingest
```

**Request body:**
```json
{
  "csv_dir": "../inputFiles/csv3",
  "graph_name": "hygraph",
  "batch_size": 5000,
  "schema": {
    "Station": {
      "capacity": "int",
      "lat": "float",
      "lon": "float",
      "region_id": "int"
    },
    "Trip": {
      "duration": "int",
      "distance": "float"
    }
  }
}
```

### Schema Parameter

The `schema` parameter defines type conversions for node/edge properties. Without it, all properties are loaded as strings.


**Schema structure:**
```json
{
  "LabelName": {
    "property_name": "type",
    "another_property": "type"
  }
}
```

**Response (success):**
```json
{
  "success": true,
  "graph_name": "hygraph",
  "stats": {
    "nodes_total": 100,
    "edges_total": 500,
    "measurements": 10000,
    "elapsed_seconds": 15
  }
}
```

**Response (error):**
```json
{
  "success": false,
  "error": "Connection refused"
}
```

---

## Python Example (Direct Usage)

```python
from pathlib import Path
from hygraph_core.ingest import json_to_csv, load_csv

# Field mappings for JSON → CSV conversion
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

# Schema for type conversion during ingestion
schema = {
    "Station": {
        "capacity": "int",
        "lat": "float",
        "lon": "float",
        "region_id": "int"
    },
    "Trip": {
        "duration": "int",
        "distance": "float"
    }
}

# Step 1: Convert JSON to CSV
json_to_csv(
    json_dir=Path("../inputFiles/json"),
    output_dir=Path("../inputFiles/csv3"),
    node_field_map=node_field_map,
    edge_field_map=edge_field_map
)

# Step 2: Ingest CSV into database with schema
stats = load_csv(
    csv_dir=Path("../inputFiles/csv3"),
    graph_name="hygraph",
    skip_age=False,
    schema=schema
)
print("Ingested:", stats)
```

---

## Python Example (API Usage)

```python
import requests

# Step 1: Convert JSON to CSV
resp = requests.post("http://localhost:8001/api/convert", json={
    "input_dir": "../inputFiles/json",
    "output_dir": "../inputFiles/csv3",
    "node_field_map": {
        "oid": "station_id",
        "start_time": "start",
        "end_time": "end"
    },
    "edge_field_map": {
        "source_id": "from",
        "target_id": "to",
        "start_time": "start",
        "end_time": "end"
    }
})
result = resp.json()
if result["success"]:
    print("Converted:", result["stats"])
else:
    print("Error:", result["error"])

# Step 2: Ingest to HyGraph with schema
resp = requests.post("http://localhost:8001/api/ingest", json={
    "csv_dir": "../inputFiles/csv3",
    "graph_name": "hygraph",
    "schema": {
        "Station": {
            "capacity": "int",
            "lat": "float",
            "lon": "float",
            "region_id": "int"
        }
    }
})
result = resp.json()
if result["success"]:
    print("Ingested:", result["stats"])
else:
    print("Error:", result["error"])
```

---

## cURL Examples

**Convert:**
```bash
curl -X POST http://localhost:8001/api/convert \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "../inputFiles/json",
    "output_dir": "../inputFiles/csv3",
    "node_field_map": {"oid": "station_id", "start_time": "start", "end_time": "end"},
    "edge_field_map": {"source_id": "from", "target_id": "to", "start_time": "start", "end_time": "end"}
  }'
```

**Ingest with schema:**
```bash
curl -X POST http://localhost:8001/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "csv_dir": "../inputFiles/csv3",
    "graph_name": "hygraph",
    "schema": {
      "Station": {"capacity": "int", "lat": "float", "lon": "float", "region_id": "int"}
    }
  }'
```

---

## Expected Folder Structure

**Input (JSON):**
```
input_dir/
  nodes/
    Station.json
  edges/
    Trip.json
```

**Output (CSV):**
```
output_dir/
  nodes/
    Station.csv
  edges/
    Trip.csv
  measurements.csv
```

---


