# The script transforms RDBMS schemas to NoSQL using a GNN:

# Input Data: Defines TPC-H tables, queries, FKs, and sizes.
# Graph Construction: Builds a graph (build_schema_graph) with nodes (tables) and edges (relationships) plus features.
# Model Definition: Defines the GNN (SchemaGNN) for prediction.
# Training: Prepares data, trains the GNN on labels (embed vs. reference).
# Schema Generation: Uses trained model to predict and apply rules.
# Validation & Visualization: Checks schema, saves JSON, and plots a graph.
# Main Execution: Runs everything in sequence.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
from collections import defaultdict
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple

# === INPUT DATA ===
tpch_query_usage = {
    "Lineitem": [1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 17, 18, 19, 20],
    "Orders": [3, 4, 5, 7, 8, 9, 10, 12, 13, 17, 20, 21],
    "Customer": [3, 5, 7, 8, 10, 12, 17, 21],
    "Supplier": [2, 5, 7, 8, 9, 11, 15, 19, 20],
    "Part": [2, 8, 9, 14, 15, 16, 18, 19],
    "PartSupp": [2, 9, 11, 16, 19],
    "Region": [2, 5, 8],
    "Nation": [2, 5, 7, 8, 9, 10, 11, 19, 20],
}

foreign_keys = {
    "Lineitem": ["Orders", "Part", "PartSupp", "Supplier"],
    "Orders": ["Customer"],
    "Customer": ["Nation"],
    "Supplier": ["Nation", "Region"],
    "PartSupp": ["Part", "Supplier"],
    "Nation": ["Region"]
}

table_sizes = {
    "Lineitem": 6_000_000,
    "Orders": 1_500_000,
    "Customer": 150_000,
    "Part": 200_000,
    "Supplier": 10_000,
    "PartSupp": 800_000,
    "Nation": 25,
    "Region": 5
}

# === GRAPH CONSTRUCTION ===
def build_schema_graph():
    G = nx.DiGraph()
    
    # Normalize features
    max_size = max(table_sizes.values()) # Gets the largest table size (Lineitem: 6M).
    max_queries = max(len(q) for q in tpch_query_usage.values())  # Gets max query count (Lineitem: 16).
    
    # Add nodes with normalized features
        # Add nodes with normalized features: Loops through tables, adds them as nodes with scaled features.

    for table in tpch_query_usage.keys(): # For each table (e.g., "Lineitem").
        G.add_node(table,  # Add node with name.
                  size=np.log10(table_sizes[table] + 1) / np.log10(max_size + 1), # Normalizes size (log to handle large numbers, +1 avoids log(0)).
                  query_freq=len(tpch_query_usage[table])/max_queries,# Normalizes query frequency (e.g., 16/16=1 for Lineitem).
                  is_dimension=int(table in ['Nation', 'Region']))  # Binary flag (1 for small dimension tables, 0 otherwise).
    
    # Add edges with features
      # Add edges with features: Loops through FKs, adds directed edges with features.
    for child, parents in foreign_keys.items():  # For each child table and its parents (e.g., Lineitem → ["Orders", "Part"]).
        for parent in parents: # For each parent.
            overlap = len(set(tpch_query_usage[child]) & set(tpch_query_usage[parent])) # Counts shared queries (intersection).
            total = len(set(tpch_query_usage[child]) | set(tpch_query_usage[parent]))  # Counts total unique queries (union).
            G.add_edge(child, parent,  # Adds directed edge (child → parent).
                      query_overlap=overlap/max(1, total), # Jaccard similarity (overlap ratio, e.g., 10/20=0.5).
                      size_ratio=np.log10((table_sizes[child] + 1)/(table_sizes[parent] + 1))) # Log size ratio (child/parent, +1 avoids division by zero).
    # Builds the graph → Converts to PyTorch Geometric Data (tensors for GNN).
    return G # Returns the graph (used in `prepare_data` for GNN input).

# === GNN MODEL ===
 
class SchemaGNN(nn.Module):  # Inherits from PyTorch's nn.Module for neural network functionality.
    def __init__(self, hidden_channels=64):  # Constructor; hidden_channels (default 64) controls layer size.
        super().__init__()  # Calls parent class initializer.
        self.conv1 = GCNConv(3, hidden_channels)  # First GCN layer: 3 input features (node size, query_freq, is_dimension) → hidden_channels.
        self.conv2 = GCNConv(hidden_channels, hidden_channels)  # Second GCN layer: hidden_channels → hidden_channels (refines embeddings).
        self.dropout = nn.Dropout(0.3)  # Dropout layer: Randomly drops 30% of neurons during training to prevent overfitting.
        self.edge_classifier = nn.Linear(hidden_channels*2 + 2, 1)  # Linear classifier: hidden_channels*2 (src+dst node embeddings) + 2 (edge features) → 1 output (probability).
        
    def forward(self, x, edge_index, edge_attr):  # Forward pass: Defines how data flows through the network.
        # Node embeddings: Processes node features through GCN layers.
        x = self.conv1(x, edge_index).relu()  # Applies first GCN conv, then ReLU activation (introduces non-linearity).
        x = self.dropout(x)  # Applies dropout to x.
        x = self.conv2(x, edge_index)  # Applies second GCN conv (no activation here, as it's for embeddings).
        
        # Edge classification: Predicts for each edge.
        src, dst = edge_index[0], edge_index[1]  # Gets source and destination node indices for edges.
        edge_features = torch.cat([x[src], x[dst], edge_attr], dim=1)  # Concatenates src node embedding, dst node embedding, and edge attributes.
        return torch.sigmoid(self.edge_classifier(edge_features))  # Applies linear classifier, then sigmoid (outputs 0-1 probability for embed).
# === TRAINING ===
def prepare_data() -> Tuple[Data, Dict[str, int], torch.Tensor]:
    G = build_schema_graph()  # Calls build_schema_graph() to create NetworkX graph with nodes/edges/features.
    
    node_map = {node: i for i, node in enumerate(G.nodes())}  # Creates a mapping from table names to indices (e.g., "Lineitem": 0). Needed for tensor conversion.
    
    # Node features: Extracts and stacks node features into a tensor.
    x = torch.tensor([  # Creates a 2D tensor for node features (shape: [num_nodes, 3]).
        [G.nodes[n]['size'], G.nodes[n]['query_freq'], G.nodes[n]['is_dimension']]  # For each node, gets normalized size, query freq, and dimension flag.
        for n in G.nodes()  # Loops through all nodes.
    ], dtype=torch.float)  # Converts to float tensor for GNN input.
    
    # Edge indices and features: Converts edges to tensors.
    edge_index = torch.tensor([  # Creates edge index tensor (shape: [2, num_edges]) for PyTorch Geometric.
        [node_map[src], node_map[dst]] for src, dst in G.edges()  # Maps node names to indices for each edge.
    ]).t().contiguous()  # Transposes and makes contiguous (required format: [src_indices, dst_indices]).
    
    edge_attr = torch.tensor([  # Creates edge attribute tensor (shape: [num_edges, 2]).
        [G.edges[e]['query_overlap'], G.edges[e]['size_ratio']]  # For each edge, gets overlap and size ratio.
        for e in G.edges  # Loops through all edges.
    ], dtype=torch.float)  # Converts to float tensor.
    
    # IMPROVED LABEL GENERATION: Creates ground-truth labels for training (0=reference, 1=embed).
    labels = []  # Empty list for labels.
    for e in G.edges:  # Loops through each edge.
        child, parent = e  # Unpacks child and parent nodes.
        overlap = G.edges[e]['query_overlap']  # Gets overlap feature.
        size_ratio = G.edges[e]['size_ratio']  # Gets size ratio feature.
        
        # More sophisticated labeling strategy: Applies heuristic rules for labels.
        if parent in ['Nation', 'Region']:  # If parent is a small dimension table.
            label = 1 if size_ratio < 0 else 0  # Embed if child is smaller (size_ratio < 0 means child < parent).
        elif overlap > 0.6 and size_ratio < -1:  # High overlap and child much smaller.
            label = 1  # Embed.
        elif table_sizes[child] < 100000 and overlap > 0.4:  # Small child with decent overlap.
            label = 1  # Embed.
        else:  # Default case.
            label = 0  # Reference.
        labels.append(label)  # Adds label to list.
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), node_map, torch.tensor(labels, dtype=torch.float).view(-1, 1)  # Returns Data object, node map, and labels tensor.

def train_model() -> Tuple[SchemaGNN, Dict[str, int]]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Sets device to GPU if available, else CPU.
    print(f"Using device: {device}")  # Prints device for debugging.
    
    data, node_map, labels = prepare_data()  # Calls prepare_data() to get Data object, node map, and labels.
    
    # Move tensors to device with proper null checks: Ensures tensors are on the correct device.
    if data.x is not None:  # Checks if node features exist.
        data.x = data.x.to(device)  # Moves to device.
    if data.edge_index is not None:  # Checks edge indices.
        data.edge_index = data.edge_index.to(device)
    if data.edge_attr is not None:  # Checks edge attributes.
        data.edge_attr = data.edge_attr.to(device)
    if labels is not None:  # Checks labels.
        labels = labels.to(device)
    
    model = SchemaGNN().to(device)  # Initializes SchemaGNN model and moves to device.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)  # Adam optimizer: lr=0.01 (learning rate), weight_decay=1e-4 (L2 regularization to prevent overfitting).
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss: Suitable for 0-1 classification (embed/reference).
    
    print("Starting training...")  # Prints start message.
    best_loss = float('inf')  # Tracks best loss for monitoring.
    
    for epoch in range(300):  # Training loop: 300 epochs.
        model.train()  # Sets model to training mode (enables dropout, etc.).
        optimizer.zero_grad()  # Resets gradients from previous step.
        
        # Ensure tensors exist before using them: Safety checks.
        if data.x is not None and data.edge_index is not None and data.edge_attr is not None:  # Checks all required tensors.
            out = model(data.x, data.edge_index, data.edge_attr)  # Forward pass: Gets predictions from model.
            if labels is not None:  # Checks labels.
                loss = criterion(out, labels)  # Computes loss between predictions and labels.
                loss.backward()  # Backpropagation: Computes gradients.
                optimizer.step()  # Updates model weights.
                
                if loss.item() < best_loss:  # If current loss is better.
                    best_loss = loss.item()  # Updates best loss.
                
                if epoch % 30 == 0:  # Every 30 epochs.
                    print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Best Loss: {best_loss:.4f}')  # Prints progress.
    
    print("Training complete!")  # Prints completion.
    return model, node_map  # Returns trained model and node map.

# === SCHEMA GENERATION ===
def generate_schema(model: SchemaGNN, node_map: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
    # Get device from model parameters
    device = next(iter(model.parameters())).device
    model.eval()
    data, _, _ = prepare_data()
    
    # Move data to same device as model with null checks
    if data.x is not None:
        data.x = data.x.to(device)
    if data.edge_index is not None:
        data.edge_index = data.edge_index.to(device)
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr.to(device)
    
    with torch.no_grad():
        if data.x is not None and data.edge_index is not None and data.edge_attr is not None:
            predictions = model(data.x, data.edge_index, data.edge_attr)
        else:
            raise ValueError("Data tensors are None")
    
    # Reverse node mapping
    rev_map = {v: k for k, v in node_map.items()}
    
    collections = defaultdict(lambda: {"embed": [], "reference": []})
    
    if data.edge_index is not None:
        for i, (src, dst) in enumerate(data.edge_index.t().tolist()):
            child = rev_map[src]
            parent = rev_map[dst]
            
            if predictions[i].item() > 0.5:
                collections[parent]["embed"].append(child)
            else:
                collections[parent]["reference"].append(child)
    
    # Apply domain rules
    collections = apply_tpch_rules(collections)
    
    # Add standalone tables
    for table in tpch_query_usage:
        if table not in collections and not any(table in v["embed"] for v in collections.values()):
            collections[table] = {"embed": [], "reference": []}
    
    return dict(collections)

def apply_tpch_rules(collections):
    # Convert to sets for clean operations
    for collection in collections.values():
        collection["embed"] = set(collection["embed"])
        collection["reference"] = set(collection["reference"])
    
    # 1. Force critical embeddings for performance
    collections["Orders"]["embed"] = {"Lineitem"}
    collections["Orders"]["reference"].discard("Lineitem")
    
    # 2. Optimize dimension tables
    collections["Customer"]["embed"].add("Nation")
    collections["Customer"]["reference"].discard("Nation")
    
    # Remove Nation from other collections to avoid duplication
    for coll_name in ["Supplier", "Region"]:
        if "Nation" in collections[coll_name]["embed"]:
            collections[coll_name]["embed"].remove("Nation")
            collections[coll_name]["reference"].add("Nation")
    
    # 3. Handle Part-PartSupp relationships
    if "PartSupp" in collections.get("Part", {}).get("reference", set()):
        collections["Part"]["reference"].remove("PartSupp")
    collections["PartSupp"]["reference"].add("Part")
    
    # 4. Handle Supplier-PartSupp (prevent large embeds)
    if "PartSupp" in collections.get("Supplier", {}).get("embed", set()):
        collections["Supplier"]["embed"].remove("PartSupp")
    collections["PartSupp"]["reference"].add("Supplier")
    
    # Convert back to lists
    for collection in collections.values():
        collection["embed"] = list(collection["embed"])
        collection["reference"] = list(collection["reference"])
    
    return collections

def visualize_schema(schema):
    plt.figure(figsize=(16, 12))
    G = nx.DiGraph()
    
    # Add nodes
    for collection in schema:
        G.add_node(collection, size=table_sizes.get(collection, 1))
    
    # Add edges
    for parent, rels in schema.items():
        for child in rels["embed"]:
            G.add_edge(parent, child, relationship="embed", color="green", style="dashed")
        for child in rels["reference"]:
            G.add_edge(parent, child, relationship="reference", color="blue", style="solid")
    
    # Layout
    layers = {
        'Region': 0, 'Nation': 1, 'Supplier': 2, 'Customer': 2,
        'Part': 3, 'PartSupp': 3, 'Orders': 4, 'Lineitem': 5
    }
    pos = {}
    layer_counts = defaultdict(int)
    
    for node in G.nodes():
        layer = layers.get(node, 3)
        pos[node] = (layer_counts[layer], -layer)
        layer_counts[layer] += 1
    
    # Draw nodes
    node_sizes = [max(100, int(table_sizes[n] * 0.0001)) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,  # type: ignore
                          node_color="lightblue", alpha=0.8)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")
    
    # Draw edges by type
    embed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relationship') == 'embed']
    ref_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relationship') == 'reference']
    
    if embed_edges:
        nx.draw_networkx_edges(G, pos, edgelist=embed_edges,
                             edge_color="green", style="dashed", width=2, alpha=0.7)
    if ref_edges:
        nx.draw_networkx_edges(G, pos, edgelist=ref_edges,
                             edge_color="blue", style="solid", width=2, alpha=0.7)
    
    # Legend
    plt.plot([], [], color="green", linestyle="dashed", linewidth=3, label="Embedding")
    plt.plot([], [], color="blue", linewidth=3, label="Reference")
    plt.legend(loc='upper right', fontsize=12)
    
    plt.title("NoSQL Schema Transformation (RDBMS → NoSQL)", fontsize=16, pad=20)
    plt.axis("off")
    plt.tight_layout()
    
    # Save
    plt.savefig("nosql_schema_transformation.png", dpi=300, bbox_inches='tight')
    plt.show()

def validate_schema(collections):
    issues = []
    embedding_counts = defaultdict(int)
    
    for parent, rels in collections.items():
        for child in rels["embed"]:
            embedding_counts[child] += 1
            if table_sizes[child] > table_sizes[parent] * 10:
                issues.append(f"Size warning: Large table {child} embedded in {parent}")
    
    for table, count in embedding_counts.items():
        if count > 1:
            issues.append(f"Embedding conflict: {table} embedded in {count} collections")
    
    # Check for orphaned tables
    all_referenced = set()
    for rels in collections.values():
        all_referenced.update(rels["embed"])
        all_referenced.update(rels["reference"])
    
    orphaned = set(tpch_query_usage.keys()) - all_referenced - set(collections.keys())
    if orphaned:
        issues.append(f"Orphaned tables: {orphaned}")
    
    if issues:
        print("\n⚠️ Schema Validation Issues:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("\n✅ Schema validated successfully - no issues found!")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("🚀 Starting RDBMS to NoSQL Schema Transformation")
    print("=" * 60)
    
    # 1. Train the GNN model
    print("\n📊 Training Graph Neural Network...")
    model, node_map = train_model()
    
    # 2. Generate NoSQL schema
    print("\n🔄 Generating NoSQL Schema...")
    schema = generate_schema(model, node_map)
    
    # 3. Display results
    print("\n📋 Generated NoSQL Schema:")
    print("=" * 40)
    for collection, relations in schema.items():
        if relations["embed"] or relations["reference"]:
            print(f"\n🗂️  Collection: {collection}")
            if relations["embed"]:
                print(f"   📦 Embedded: {relations['embed']}")
            if relations["reference"]:
                print(f"   🔗 Referenced: {relations['reference']}")
    
    # 4. Validate schema
    print("\n🔍 Validating Schema...")
    validate_schema(schema)
    
    # 5. Save results
    with open("nosql_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    print("\n💾 Schema saved to 'nosql_schema.json'")
    
    # 6. Visualize
    print("\n📊 Generating visualization...")
    visualize_schema(schema)
    print("\n✅ Process completed successfully!")



# 1. Overall Approach and Purpose

# Approach: Uses a Graph Neural Network (GNN) to model RDBMS schemas as graphs (tables as nodes, FKs as edges) and predict "embed" vs. "reference" decisions. It leverages graph structure to learn relational patterns, trained on heuristic labels, and refines with rules.
# Purpose: Transforms relational schemas to NoSQL (e.g., MongoDB) by optimizing for query performance and data locality. GNNs capture "neighborhood effects" (e.g., embedding based on connected tables), making it better for graph-like data than flat ML. It produces a valid, efficient NoSQL schema with embeds/references.
# Key Idea: Treats schema as a graph, uses GNN to predict edge labels (embed/reference), applies domain rules, validates, and visualizes.
# Why This Approach?: Handles relational dependencies (FKs) better than non-graph methods; scalable for larger schemas; combines learning with rules for accuracy.
# Flow: Input data → Graph build → Data prep → Train GNN → Generate schema → Apply rules → Validate → Visualize.


# 2. Explanation of All Functions/Methods
# build_schema_graph: Constructs a directed graph from TPC-H schema. Nodes are tables with features (size, query freq, dimension flag).
  #  Edges are FKs with features (query overlap, size ratio). Returns NetworkX graph.

# SchemaGNN: Defines a GNN model with two GCN layers and dropout. 
  # Takes node features, edge indices, and edge attributes. 
  # Outputs probabilities for each edge being "embed" (1) or "reference" (0).
  #  forward: Implements forward pass: processes node features through GCN layers, concatenates src/dst embeddings with edge features, classifies edges with sigmoid output.
   # def __init__: Initializes GNN layers, dropout, and edge classifier.
       #This is the constructor of the SchemaGNN class. It's called when you create the model (e.g., model = SchemaGNN()). It sets up the "brain" structure by defining the layers and components.
       #Purpose: Builds the neural network architecture. It defines the layers (like assembling parts of a machine) so the model knows how to process data. Without this, the model wouldn't know what to do with inputs.

       #How it works :- Analogy: Like building a factory—conv1 and conv2 are assembly lines for processing parts (nodes/edges), dropout is a quality check to avoid defects, and edge_classifier is the final inspector that decides the product (embed/reference).
       #This function doesn't return anything (it's just setup). But it creates the model's "structure," which is used later in forward. The result is a ready-to-use GNN object with defined layers.
    

    # def_forward : - This is the core processing method. It's called automatically during training/inference (e.g., model(data.x, data.edge_index, data.edge_attr)). It defines how data flows through the model to produce predictions.
      #Purpose: Processes input data (node features, edge connections, edge features) through the layers to generate predictions. It's like running data through the "brain" to get answers.  
      #Analogy: Like a team meeting—nodes "discuss" with neighbors via GCN layers (building smarter representations), then the classifier "votes" on each relationship (edge) based on combined info.
      #Achievement: Predictions for every edge in the graph. E.g., for Lineitem→Orders edge, output ~0.9 (high probability to embed). This is used in generate_schema to build the NoSQL collections dict.

# prepare_data: Builds the schema graph, extracts node/edge features,
  # converts to PyTorch tensors.
    # Generates heuristic labels for edges (0/1) based on rules. 
    # Returns PyTorch Geometric Data object, node map, and labels tensor.

# train_model: Prepares data, initializes SchemaGNN, trains it using Adam optimizer and BCELoss for 300 epochs.
  # Moves data/model to GPU if available. Returns trained model and node map.

# generate_schema: Uses trained model to predict edge labels on the schema graph.
  # Constructs collections with "embed" and "reference" lists based on predictions.

# apply_tpch_rules: Applies domain-specific rules to refine schema (e.g., force Orders to embed Lineitem).
  # Ensures dimension tables are handled properly, avoids large embeds. Returns refined collections.

# visualize_schema: Visualizes the generated schema as a directed graph using NetworkX and Matplotlib.
  # Nodes are collections, edges are embeds/references with different styles/colors. Saves as PNG

# validate_schema: Validates the generated schema for issues like large embeds, embedding conflicts, orphaned tables.
  # Prints warnings if issues are found.

# Main Execution Block: Runs the entire process:
  # Trains the GNN, generates schema, displays results, validates, saves to JSON
  # and visualizes the schema transformation.
  