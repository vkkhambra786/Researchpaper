# # Supervised Learning for TPC-H NoSQL Schema Optimization
# # This script enhances the TPC-H NoSQL schema optimization process using supervised learning techniques.
 
# # This code is designed to be run in an environment with the necessary libraries installed.
# # Requirements:
# # - pandas
# # - numpy
# # - scikit-learn
# # - json
# # - collections
# # This code is a fixed version of the original TPC-H schema optimization script, addressing issues with embedding and relationships based on TPC-H best practices.
# ## The code is structured to first generate a schema based on TPC-H query usage, then validate the schema, and finally output the optimized schema along with relationship decisions.
# # Optimized TPC-H NoSQL Schema with Supervised Learning
 
# # Output is a JSON representation of the optimized schema, along with explanations of relationship decisions.
# # Output schema transformation from RDBMS to nosql  using supervised learning techniques.
 
# graph TD
#     A[RDBMS Schema] -->|Parse| B[Input Data]
#     B --> C[Training Data]
#     C --> D[Model Training]
#     D --> E[Schema Generator]
#     E -->|Features| F[Embed/Reference Decisions]
#     F --> G[NoSQL Schema]
#     G --> H[Validation]
#     H -->|Valid| I[Final Schema]
#     H -->|Invalid| E
 
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score



# === Input Data ===
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

# === Schema Validation ===
 
# === Training Data Generation ===
def generate_training_data():
    features = ["size_ratio", "co_access_freq", "is_dimension", "join_depth", "decision"]
    # data = [
    #     [0.004, 1.0, 1, 1, 1],  # Nation in Customer
    #     [0.003, 1.0, 1, 1, 1],  # Region in Nation
    #     [0.25, 0.9, 0, 1, 1],   # Lineitem in Orders
    #     [400.0, 0.3, 0, 2, 0],  # Supplier in Nation
    #     [2000.0, 0.1, 0, 2, 0], # Supplier in Region
    #     [3.0, 0.4, 0, 2, 0]     # PartSupp in Part
    # ]
    data = [
        # size_ratio, co_access, is_dim, join_depth, decision
        [0.004,  1.0, 1, 1, 1],  # Tiny dimension table (Embed)
        [0.2,    0.9, 0, 1, 1],   # High co-access (Embed)
        [10.0,   0.8, 0, 1, 0],   # Large child but high co-access (Reference)
        [500.0,  0.1, 0, 2, 0],   # Huge size ratio (Reference)
        [0.001,  0.3, 1, 1, 1],   # Tiny lookup table (Embed)
        [5.0,    0.6, 0, 1, 0],   # Medium size ratio (Reference)
        [0.5,    0.7, 0, 1, 1]    # Balanced case (Embed)
    ]
    return pd.DataFrame(data, columns=features)

# === Model Training ===
def train_model():
    train_df = generate_training_data()
    X = train_df.iloc[:, :-1]
    y = train_df.iloc[:, -1]
    
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    
    print("Model Evaluation:")
    print(classification_report(y, model.predict(X)))
    
    return model

# === Schema Generation ===
def generate_schema_ml(model):
    collections = defaultdict(lambda: {"embed": [], "reference": []})
    
    # Calculate co-access frequencies
    co_access = defaultdict(dict)
    for t1 in tpch_query_usage:
        for t2 in tpch_query_usage:
            shared = len(set(tpch_query_usage[t1]) & set(tpch_query_usage[t2]))
            co_access[t1][t2] = shared / len(tpch_query_usage[t1])
    
    # TPC-H mandatory patterns with strict hierarchy
    collections["Orders"]["embed"] = ["Lineitem"]
    collections["Customer"]["embed"] = ["Nation"]
    collections["Nation"]["embed"] = ["Region"]
    collections["Orders"]["reference"] = ["Customer"]  # Only Orders -> Customer
    
    # Track embedded tables to prevent duplicates
    embedded_tables = set(["Lineitem", "Nation", "Region"])
    
    # Process remaining relationships with circular reference prevention
    for child, parents in foreign_keys.items():
        for parent in parents:
            # Skip if already processed or would create circular reference
            if (child in collections[parent]["embed"] or 
                child in collections[parent]["reference"] or
                (parent == "Customer" and child == "Orders")):  # Prevent circular reference
                continue
                
            size_ratio = table_sizes[child] / table_sizes[parent]
            
            if (size_ratio > 100 or 
                child in embedded_tables):
                collections[parent]["reference"].append(child)
                continue
                
            co_access_score = co_access.get(parent, {}).get(child, 0.0)
            features = [[
                size_ratio,
                co_access_score,
                1 if child in ["Nation", "Region"] else 0,
                1
            ]]
            
            if bool(model.predict(features)[0]):
                collections[parent]["embed"].append(child)
                embedded_tables.add(child)
            else:
                collections[parent]["reference"].append(child)
    
    validate_schema(collections)
    return collections

def validate_schema(collections):
    issues = []
    embedding_counts = defaultdict(int)
    
    # Check for duplicates and circular references
    for parent, rels in collections.items():
        # Remove duplicates
        collections[parent]["embed"] = list(set(collections[parent]["embed"]))
        collections[parent]["reference"] = list(set(collections[parent]["reference"]))
        
        for child in rels["embed"]:
            embedding_counts[child] += 1
    
    # Check for circular references
    if ("Customer" in collections["Orders"]["reference"] and 
        "Orders" in collections["Customer"]["reference"]):
        issues.append("Circular reference between Orders and Customer")
    
    # Check for duplicate embeddings
    for table, count in embedding_counts.items():
        if count > 1:
            issues.append(f"Duplicate embedding: {table} embedded in {count} collections")
    
    if not issues:
        print("✅ Schema validated with no issues")
    else:
        print("Validation Issues:")
        for issue in issues: print(f"⚠️ {issue}")
# === Visualization ===
def plot_schema(collections):
    plt.figure(figsize=(12, 8))

    plt.title("Optimized TPC-H NoSQL Schema Visualization", fontsize=16)
    positions = {
        "Orders": (2, 5), "Lineitem": (2, 4),
        "Customer": (4, 5), "Nation": (4, 4), "Region": (4, 3),
        "Part": (0, 4), "PartSupp": (0, 3), "Supplier": (6, 4)
    }
    
    # Draw edges
    for parent, rels in collections.items():
        for child in rels["embed"]:
            plt.plot([positions[parent][0], positions[child][0]], 
                    [positions[parent][1], positions[child][1]], 
                    'g-', linewidth=2)
            mid_x = (positions[parent][0] + positions[child][0])/2
            mid_y = (positions[parent][1] + positions[child][1])/2
            plt.text(mid_x, mid_y, "Embed", color='green', ha='center')
            
        for child in rels["reference"]:
            plt.plot([positions[parent][0], positions[child][0]], 
                    [positions[parent][1], positions[child][1]], 
                    'r--', linewidth=1)
            mid_x = (positions[parent][0] + positions[child][0])/2
            mid_y = (positions[parent][1] + positions[child][1])/2
            plt.text(mid_x, mid_y, "Ref", color='red', ha='center')
    
    # Draw nodes
    for node, pos in positions.items():
        plt.text(pos[0], pos[1], node, ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    
    plt.title("Optimized TPC-H NoSQL Schema\n(Green = Embed, Red = Reference)", pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# === Main Execution ===
if __name__ == "__main__":
    model = train_model()
    collections = generate_schema_ml(model)
    
    print("\nOptimized Schema:")
    print(json.dumps(collections, indent=2))
    
    print("\nRelationship Decisions:")
    for parent, rels in collections.items():
        for child in rels["embed"]:
            print(f"{parent:10} EMBEDS    {child}")
        for child in rels["reference"]:
            print(f"{parent:10} REFERENCES {child}")
    
    plot_schema(collections)