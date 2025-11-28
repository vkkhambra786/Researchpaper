import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import networkx as nx
import json
from collections import defaultdict
from matplotlib.lines import Line2D

# === STEP 1: Input Data ===
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

# === STEP 2: Enhanced Feature Engineering ===
def generate_features(schema_usage, fk_relations):
    #  Its primary purpose is to transform raw schema and
    #  workload data into a structured, machine learning-ready dataset
    # This dataset is used to train a supervised ML model (Random Forest) to predict whether to "embed" or "reference" relationships in a NoSQL schema.
    data = []
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
    
    # for table, queries in schema_usage.items():
    #     query_set = set(queries)
    #     fks = fk_relations.get(table, [])
    #     for parent in fks:
    #         parent_queries = set(schema_usage.get(parent, []))
    #         overlap = len(query_set & parent_queries) / max(len(query_set | parent_queries), 1)
            
    #         data.append({
    #             "table": table,
    #             "parent": parent,
    #             "query_overlap": overlap,
    #             "size_ratio": np.log10(table_sizes[table] / table_sizes[parent]),
    #             "is_lineitem": int(table == "Lineitem"),
    #             "is_dimension": int(parent in ["Nation", "Region"]),
    #             "join_frequency": len(query_set & parent_queries),
    #             "access_frequency": len(query_set),
    #             "parent_access_frequency": len(parent_queries),
    #             "is_fact_table": int(table in ["Lineitem", "Orders", "PartSupp"]),
    #             "cardinality": 1 if table_sizes[table] < table_sizes[parent] else 0,
    #             "label": "embed" if overlap > 0.6 else "reference"
    #         })
    # return pd.DataFrame(data)
    for table, queries in schema_usage.items():  # Loop 1: Through each table and its queries
       query_set = set(queries)  # Convert queries to a set for fast operations
       fks = fk_relations.get(table, [])  # Get foreign keys (parents) for this table
       for parent in fks:  # Loop 2: Through each parent of this table
           parent_queries = set(schema_usage.get(parent, []))  # Get parent's queries as a set
        
        # Calculate query overlap (Jaccard similarity)
           overlap = len(query_set & parent_queries) / max(len(query_set | parent_queries), 1)
        
        # Append a dictionary (row) to the data list
           data.append({
            "table": table,  # Child table name (e.g., "Lineitem")
            "parent": parent,  # Parent table name (e.g., "Orders")
            "query_overlap": overlap,  # Feature: How much queries overlap (0-1)
            "size_ratio": np.log10(table_sizes[table] / table_sizes[parent]),  # Feature: Log of size ratio
            "is_lineitem": int(table == "Lineitem"),  # Feature: 1 if child is Lineitem, else 0
            "is_dimension": int(parent in ["Nation", "Region"]),  # Feature: 1 if parent is small dim, else 0
            "join_frequency": len(query_set & parent_queries),  # Feature: Count of shared queries
            "access_frequency": len(query_set),  # Feature: How often child is queried
            "parent_access_frequency": len(parent_queries),  # Feature: How often parent is queried
            "is_fact_table": int(table in ["Lineitem", "Orders", "PartSupp"]),  # Feature: 1 if child is fact, else 0
            "cardinality": 1 if table_sizes[table] < table_sizes[parent] else 0,  # Feature: 1 if child smaller, else 0
            "label": "embed" if overlap > 0.6 else "reference"  # Label: Correct answer based on overlap
           })
    return pd.DataFrame(data)  # Convert list of dicts to DataFrame

df = generate_features(tpch_query_usage, foreign_keys)

# === STEP 3: Model Training with Enhanced Features ===
label_encoders = {}    #Empty dict to hold LabelEncoder objects (one per column).
df_encoded = df.copy()  #Creates a duplicate of df (your feature DataFrame) for safe modifications.
# copy() is a Pandas method for deep copying. This is setup—nothing printed yet.
for col in ["table", "parent", "label"]:
    #  Converts categorical columns (text) to numbers (e.g., "Orders" → 0, "embed" → 1).
    le = LabelEncoder()  #Creates a new encoder object.
    df_encoded[col] = le.fit_transform(df[col])  # Learns mappings from the original df (e.g., "Orders" → 0, "Lineitem" → 1) and applies them to df_encoded
    label_encoders[col] = le # Learns mappings from the original df (e.g., "Orders" → 0, "Lineitem" → 1) and applies them to df_encoded

X = df_encoded.drop(columns=["label"]) #Removes the "label" column, leaving features (e.g., query_overlap, size_ratio).
# drop() is a Pandas method. X is a DataFrame/matrix; y is a Series/array
y = df_encoded["label"] #Keeps only the "label" column (0 for "reference", 1 for "embed").

scaler = StandardScaler() #Creates a scaler object.
# StandardScaler() adjusts features so big values (e.g., sizes) don't overpower small ones (e.g., overlaps).
#StandardScaler().fit_transform() from sklearn.preprocessing. Uses Z-score: (value - mean) / std_dev. X_scaled is a NumPy array.
X_scaled = scaler.fit_transform(X) #Learns scaling (mean/std) from X and applies it. Example: query_overlap 0.5 becomes ~0.2.
# (X = features, without labels).

# Enhanced evaluation with cross-validation
clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced') #Creates the model.
# n_estimators=150: Uses 150 decision trees (more = better but slower).
# random_state=42: Ensures reproducible results (same random seed).
# class_weight='balanced': Handles imbalanced labels (e.g., if "embed" is rarer, it gives it more weight).
cv_scores = cross_val_score(clf, X_scaled, y, cv=5) # Runs 5-fold cross-validation.
print(f"\nCross-Validation Accuracy: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores):.2f})")

# Final model training
clf.fit(X_scaled, y) #Learns patterns from scaled features (X_scaled) to predict labels (y).

# Feature importance analysis
feature_importances = pd.DataFrame({
    'feature': df_encoded.drop(columns=["label"]).columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False) # Builds a DataFrame with feature names and importance scores (0-1, sum to 1).
# : Creates and prints a table showing which features mattered most to the model.6
print("\nFeature Importances:")
print(feature_importances)

# === STEP 4: Schema Generation with Enhanced Optimizations ===
# This step takes the trained machine learning model from Step 3 and uses it to build the actual NoSQL schema (a dictionary of "collections" with embeds and references). 
# Make Predictions Using the Trained Model
df_encoded["predicted_label"] = clf.predict(X_scaled)
#  Feeds the scaled features (X_scaled) into the model. The model outputs numbers (0 for "reference", 1 for "embed") based on patterns it learned (e.g., high query_overlap → embed).
# df_encoded["predicted_label"] = ...: Adds a new column to df_encoded with these predictions.

#  Convert Predictions Back to Text
df["predicted_label"] = df_encoded["predicted_label"].map({0: "reference", 1: "embed"})
# df_encoded["predicted_label"].map({0: "reference", 1: "embed"}): Uses a mapping dictionary to replace 0 with "reference" and 1 with "embed".
# df["predicted_label"] = ...: Adds the text predictions to the original df (not df_encoded).

#  Initialize the Schema Dictionary
collections = defaultdict(lambda: {"embed": [], "reference": []})
# defaultdict(lambda: {"embed": [], "reference": []}): A special dict that automatically creates a sub-dict {"embed": [], "reference": []} for any new key (parent table).
# No initial keys yet—it's empty and ready to be filled.

for _, row in df.iterrows():
    parent = row["parent"]
    child = row["table"]
    label = row["predicted_label"]
    collections[parent][label].append(child)
#     for _, row in df.iterrows(): Loops through each row in df (each relationship). _ ignores the row index; row is a Pandas Series with columns like "table", "parent", "predicted_label".
# parent = row["parent"]: Gets the parent table (e.g., "Orders").
# child = row["table"]: Gets the child table (e.g., "Lineitem").
# label = row["predicted_label"]: Gets the prediction (e.g., "embed").
# collections[parent][label].append(child): Adds the child to the appropriate list in the parent's sub-dict.
# If parent doesn't exist, defaultdict creates it with {"embed": [], "reference": []}.
# Example: For "Orders" as parent, "Lineitem" as child, "embed" as label → collections["Orders"]["embed"].append("Lineitem").
def resolve_embedding_conflicts(collections):
#     embedding_priority: A dict defining preferred parents for specific tables (e.g., Lineitem only in Orders).
# Loop: For each table in priority, find where it's currently embedded (current_embeddings).
# If a parent isn't preferred, move the table from "embed" to "reference" in that parent.
    embedding_priority = {
    "Lineitem": ["Orders"],  # Lineitem can only be embedded in Orders
    "PartSupp": ["Part"],   # PartSupp only in Part
    "Supplier": ["Nation"]  # Supplier only in Nation
    }
    
    for table, preferred_parents in embedding_priority.items(): # Loop through each table with embedding priorities
        current_embeddings = [p for p in collections if table in collections[p]["embed"]] # Find all parents currently embedding this table
        for parent in current_embeddings: # Loop through these parents
            if parent not in preferred_parents: # If the parent is not a preferred one
                collections[parent]["embed"].remove(table) # Remove the table from "embed"
                collections[parent]["reference"].append(table) # Add it to "reference"
    return collections # Return the updated collections dict

def apply_tpch_rules(collections): # Applies TPC-H specific rules to the schema
    # This function applies domain-specific rules to the NoSQL schema generated from the machine learning model
    # Force embed Lineitem in Orders (TPC-H best practice)
    if "Lineitem" not in collections["Orders"]["embed"]: # Checks if Lineitem is already embedded in Orders
        collections["Orders"]["embed"].append("Lineitem") # If not, it adds Lineitem to Orders' embed list
    
    # Always embed small dimension tables
    for dim_table in ["Nation", "Region"]: # Loops through small dimension tables
        if dim_table in collections: # Checks if the dimension table exists in collections
            collections[dim_table]["embed"].extend(collections[dim_table]["reference"]) # Embeds all referenced tables into the dimension table
            collections[dim_table]["reference"] = [] # Clears the reference list since all are now embedded
    
    return resolve_embedding_conflicts(collections) # Ensures no conflicts after applying rules

collections = apply_tpch_rules(collections) # Applies TPC-H rules to the initial collections

# Add standalone tables with proper initialization
for table in tpch_query_usage.keys(): # Ensures all tables are included in the schema
    if table not in collections: # If a table isn't already a key in collections
        collections[table] = {"embed": [], "reference": []} # Initializes it with empty embed/reference lists

def validate_schema(collections):
    G = nx.DiGraph()
    validation_passed = True
    
    # Check for cycles
    for parent, rels in collections.items():
        for child in rels["embed"]:
            G.add_edge(parent, child)
    
    if not nx.is_directed_acyclic_graph(G):
        print("Warning: Circular embedding detected!")
        validation_passed = False
    
    # Check for duplicate embeddings
    embedding_counts = defaultdict(int)
    for rels in collections.values():
        for child in rels["embed"]:
            embedding_counts[child] += 1
    
    for table, count in embedding_counts.items():
        if count > 1:
            print(f"Warning: {table} is embedded in {count} collections!")
            validation_passed = False
    
    # Check for conflicting embed/reference
    for table in collections:
        embeds = set(collections[table]["embed"])
        refs = set(collections[table]["reference"])
        if embeds & refs:
            print(f"Error: Table {table} has conflicting embed/reference relationships")
            validation_passed = False
    
    return validation_passed

schema_valid = validate_schema(collections)
print(f"\nSchema validation {'passed' if schema_valid else 'failed'}")

# Final JSON Output
print("\nOptimized NoSQL Schema:\n")
print(json.dumps(collections, indent=2))

# === STEP 5: Fixed and Enhanced Visualization ===
def visualize_schema(collections):
    plt.figure(figsize=(14, 10))
    G = nx.DiGraph()
    
    # Define node types for all tables
    table_types = {
        "Lineitem": "fact",
        "Orders": "fact",
        "Customer": "dimension",
        "Part": "dimension",
        "Supplier": "dimension",
        "PartSupp": "fact",
        "Nation": "dimension",
        "Region": "dimension"
    }
    
    # Add all tables to the graph first with their types
    for table in collections:
        node_type = table_types.get(table, "dimension")
        G.add_node(table, type=node_type)
    
    # Add edges after all nodes exist
    for parent, rels in collections.items():
        for child in rels["embed"]:
            if child in G.nodes():  # Only add if child exists
                G.add_edge(parent, child, color='green', style='solid', label='embed')
        for child in rels["reference"]:
            if child in G.nodes():  # Only add if child exists
                G.add_edge(parent, child, color='blue', style='dashed', label='reference')
    
    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.9, iterations=50)
    
    # Draw nodes by type - now all nodes have the 'type' attribute
    fact_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'fact']
    dim_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'dimension']
    
    nx.draw_networkx_nodes(G, pos, nodelist=fact_nodes, node_color='red', node_size=2500)
    nx.draw_networkx_nodes(G, pos, nodelist=dim_nodes, node_color='lightblue', node_size=2000)
    
    # Draw edges
    solid_edges = [(u,v) for (u,v,d) in G.edges(data=True) if d['style'] == 'solid']
    dashed_edges = [(u,v) for (u,v,d) in G.edges(data=True) if d['style'] == 'dashed']
    
    nx.draw_networkx_edges(G, pos, edgelist=solid_edges, edge_color='green', style='solid', arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, edge_color='blue', style='dashed', arrows=True)
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Edge labels
    edge_labels = {(u,v): d['label'] for (u,v,d) in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Fact Tables',
               markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Dimension Tables',
               markerfacecolor='lightblue', markersize=10),
        Line2D([0], [0], color='green', lw=2, label='Embed'),
        Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Reference')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Optimized TPC-H NoSQL Schema\n(Red = Fact Tables, Blue = Dimension Tables)", pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Now safely visualize the schema
visualize_schema(collections)

# === Additional Analysis for Publication ===
def analyze_schema(collections):
    # Calculate embedding statistics
    total_embeddings = sum(len(v["embed"]) for v in collections.values())
    total_references = sum(len(v["reference"]) for v in collections.values())
    
    print("\nSchema Analysis:")
    print(f"Total embeddings: {total_embeddings}")
    print(f"Total references: {total_references}")
    print(f"Embedding/Reference ratio: {total_embeddings/max(total_references,1):.2f}")
    
    # Calculate embedding depth
    G = nx.DiGraph()
    for parent, rels in collections.items():
        for child in rels["embed"]:
            G.add_edge(parent, child)
    
    try:
        max_depth = 0
        nodes = list(G.nodes())
        for source in nodes:
            for target in nodes:
                if source != target and nx.has_path(G, source, target):
                    all_paths = list(nx.all_simple_paths(G, source, target))
                    longest_path = max((len(p) for p in all_paths), default=0)
                    max_depth = max(max_depth, longest_path)

        print("\nMaximum embedding depth:", max_depth)
    except ValueError:
        print("\nMaximum embedding depth: 0 (no embedding paths)")

analyze_schema(collections)



# 1. Embedding vs Reference (in NoSQL)
# Embedding:
# Storing related data together inside a single document/collection.
# Example: Embedding all Lineitem rows inside their parent Orders document.
# Reference:
# Storing related data in separate collections and linking them via IDs (like foreign keys in RDBMS).
# Example: An Orders document stores a reference (ID) to a Customer document.
# 2. Foreign Key
# In RDBMS, a foreign key is a field in one table that links to the primary key of another table.
# In NoSQL, this relationship can be modeled as either an embed (nesting) or a reference (storing the ID).

# Step-by-Step Code Explanation
# STEP 1: Input Data
# tpch_query_usage:
# Shows which queries use each table (for workload-aware design).
# foreign_keys:
# Defines parent-child relationships (like RDBMS foreign keys).

# STEP 2: Feature Engineering
# Function: generate_features(schema_usage, fk_relations)
# Purpose:
# For each foreign key relationship, generate a set of features that describe the relationship, such as:
# Query overlap between child and parent
# Size ratio (child/parent)
# Whether the table is a fact/dimension table
# Join/access frequency
# Cardinality (is child smaller than parent)
# Label: If query overlap is high, label as "embed", else "reference" (for supervised learning)
# Returns:
# A DataFrame where each row is a relationship with all features and a label.

# STEP 3: Model Training
# Label Encoding:
# Converts categorical columns (table, parent, label) to numbers for ML.
# Scaling:
# Standardizes features for better model performance.
# Random Forest Classifier:
# Trains a model to predict "embed" vs "reference" based on features.
# Cross-Validation:
# Evaluates model accuracy and stability.
# Feature Importance:
# Shows which features are most important for the model’s decisions.
    
#     Shows which features are most important for the model’s decisions.
# STEP 4: Schema Generation
# Prediction:
# The trained model predicts "embed" or "reference" for each relationship.
# Collections Construction:
# Builds a dictionary where each parent table lists its embedded and referenced children.
# Conflict Resolution:
# resolve_embedding_conflicts(collections):
# Ensures a table is only embedded in its most appropriate parent (avoids duplication).
# apply_tpch_rules(collections):
# Applies domain-specific rules (e.g., always embed Lineitem in Orders, always embed small dimension tables).
# Standalone Tables:
# Ensures all tables are present in the schema, even if they have no relationships.

# STEP 5: Schema Validation
# Function: validate_schema(collections)
# Purpose:
# Checks for cycles (which would cause infinite embedding)
# Checks for duplicate embeddings (a table embedded in multiple parents)
# Checks for conflicting relationships (a table both embedded and referenced by the same parent)
# Returns:
# Whether the schema is valid.

# STEP 6: Visualization
# Function: visualize_schema(collections)
# Purpose:
# Draws the schema as a graph using NetworkX and Matplotlib.
# Fact tables are red, dimension tables are blue.
# Green solid lines = embed, blue dashed lines = reference.
# Shows the structure and relationships visually.

# STEP 7: Schema Analysis
# Function: analyze_schema(collections)
# Purpose:
# Counts total embeddings and references.
# Calculates the embedding/reference ratio.
# Computes the maximum embedding depth (longest chain of embeddings).

# Summary Table of Functions
# Function Name	Purpose
# generate_features	Create ML features for each relationship
# resolve_embedding_conflicts	Prevents a table from being embedded in multiple parents
# apply_tpch_rules	Applies domain-specific rules for TPC-H schema
# validate_schema	Checks for cycles, duplicate embeddings, and conflicts
# visualize_schema	Draws the schema as a colored graph
# analyze_schema	Prints stats: total embeddings, references, and max embedding depth

# What is Embedding, Reference, Embedded, Foreign Key?
# Embedding:
# Storing related data inside the same document (NoSQL).
# Example: An order document contains an array of lineitems.

# Reference:
# Storing related data in separate documents/collections, linked by an ID.
# Example: An order document has a customer_id field pointing to a customer document.

# Embedded:
# The child data that is stored inside the parent document (the result of embedding).

# Foreign Key:
# In RDBMS, a field that links two tables. In this code, it’s the relationship you’re deciding to embed or reference in NoSQL.




# # Details Explanation
# Step-by-Step Flow of Model Training
# The code processes data in this order. Each step builds on the last.

# Step 1: Generate Features and Labels (Data Preparation)

# What Happens: The generate_features() function creates a "dataset" (Pandas DataFrame) for the ML model. Each row is one foreign key relationship (e.g., Lineitem → Orders). Columns are "features" (inputs) and "label" (correct answer).
# How It Works (Code Breakdown):
# Loops through foreign_keys (e.g., Lineitem has parents: Orders, Part, etc.).
# For each child-parent pair, calculates features:
# query_overlap: How much do their query sets overlap? (E.g., Lineitem and Orders share 10 queries out of 20 total → 0.5). Formula: len(set1 & set2) / max(len(set1 | set2), 1).
# size_ratio: Child size / parent size (logged to handle big numbers). E.g., Lineitem (6M) vs Orders (1.5M) → log(6M/1.5M) ≈ 0.6.
# join_frequency: How many shared queries? (E.g., 10).
# access_frequency: How often is child queried? (E.g., 16 for Lineitem).
# is_lineitem: 1 if child is Lineitem (special case), else 0.
# is_dimension: 1 if parent is small (Nation/Region), else 0.
# cardinality: 1 if child is smaller than parent, else 0.
# is_fact_table: 1 if child is a "fact" (big, transactional), else 0.
# Label: "embed" if overlap > 0.6 (high co-usage), else "reference". This is your "ground truth" (correct answers for training).
# Why This Step? ML needs structured data. Features capture "why" to embed (e.g., frequent queries together). Labels teach the model right/wrong.
# Output: A DataFrame (e.g., 10 rows, 12 columns). Example row: {"table": "Lineitem", "parent": "Orders", "query_overlap": 0.5, "label": "embed"}.
# Syntax/Details: Uses set operations for overlap, np.log10 for ratios, and pd.DataFrame for tabular data. This is "feature engineering"—the art of turning raw data into useful inputs.

# Step 2: Encode Categorical Data

# What Happens: Converts text (e.g., "Orders") to numbers (e.g., 0) because ML models need numbers.
# How It Works (Code Breakdown):
# label_encoders = {}: Stores encoders for later reversal.
# For columns ["table", "parent", "label"]: le = LabelEncoder(); df_encoded[col] = le.fit_transform(df[col]).
# fit_transform(): Learns mappings (e.g., "Orders" → 0, "Lineitem" → 1) and applies them.
# Example: "label" column: "embed" → 0, "reference" → 1.
# Why This Step? ML can't process strings. Encoding prevents errors.
# Output: df_encoded with numbers instead of text.
# Step 3: Scale Features

# What Happens: Normalizes numbers to a standard range (mean 0, std dev 1) so no feature dominates (e.g., big sizes overpower small overlaps).
# How It Works (Code Breakdown):
# scaler = StandardScaler(); X_scaled = scaler.fit_transform(X).
# X = df_encoded.drop(columns=["label"]): Features only (inputs).
# y = df_encoded["label"]: Labels only (outputs).
# Example: Size ratio 0.6 becomes ~0.2 after scaling.
# Why This Step? Ensures fair learning (e.g., query_overlap and size_ratio are equally important).
# Output: X_scaled (scaled features) and y (labels).

# Step 4: Train and Evaluate the Model

# What Happens: Trains Random Forest on scaled data and checks performance.
# How It Works (Code Breakdown):
# clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced'): Creates model with 150 trees, balanced classes (handles imbalanced "embed" vs "reference").
# cv_scores = cross_val_score(clf, X_scaled, y, cv=5): Splits data into 5 parts, trains/tests 5 times, returns accuracies (e.g., [0.8, 0.9, 0.7, 0.8, 0.9]).
# clf.fit(X_scaled, y): Final training on full data.
# Prints: Cross-Validation Accuracy: 0.83 (+/- 0.21) (mean ± std dev).
# Why This Step? Cross-validation ensures the model generalizes (works on new data). class_weight='balanced' fixes if one label is rare.
# Output: Trained model (clf) and accuracy stats.
# Step 5: Analyze Feature Importance

# What Happens: Shows which features influenced decisions most.
# How It Works (Code Breakdown):
# feature_importances = pd.DataFrame({'feature': columns, 'importance': clf.feature_importances_}).sort_values('importance', ascending=False).
# Example Output: query_overlap (0.30), join_frequency (0.19), etc.
# Why This Step? Explains the model (e.g., "query_overlap matters most"). Helps trust/debug.
# Output: Ranked list (e.g., query_overlap is top because co-usage drives embedding).
# Full Training Flow Summary: Raw data → Features/Labels → Encoding → Scaling → Cross-Validation Training → Feature Analysis. Result: A "trained brain" (model) that predicts "embed/reference" for new relationships.



# Part 2: Schema Generation – How It Works, Why, and Step-by-Step Flow
# Schema generation uses the trained model to build the NoSQL schema. It's like using the trained dog to fetch: Feed it new data, get predictions, and build the schema.

# Why Use Schema Generation?
# Problem: You have predictions (e.g., "embed Lineitem in Orders"), but need a complete, valid NoSQL schema (collections with embeds/references).
# Solution: Automate schema building, fix conflicts (e.g., no table embedded twice), apply rules (e.g., domain knowledge), and validate.
# Benefits: Produces a ready-to-use schema (JSON), ensures correctness, and visualizes it. Without this, predictions are useless.
# Step-by-Step Flow of Schema Generation
# Starts after training. Each step refines the schema.

# Step 1: Make Predictions

# What Happens: Use the trained model to predict "embed" or "reference" for each relationship.
# How It Works (Code Breakdown):
# df_encoded["predicted_label"] = clf.predict(X_scaled): Predicts on scaled features (e.g., 0 for "reference", 1 for "embed").
# df["predicted_label"] = df_encoded["predicted_label"].map({0: "reference", 1: "embed"}): Converts back to text.
# Why This Step? Applies learning to your data. Without predictions, no schema.
# Output: df now has "predicted_label" (e.g., "embed" for Lineitem→Orders).
                                      
                                      
# Step 2: Build Initial Collections

# What Happens: Creates a dict of "collections" (NoSQL docs) with embeds/references.
# How It Works (Code Breakdown):
# collections = defaultdict(lambda: {"embed": [], "reference": []}): Dict where keys are parents, values are lists.
# Loop: [for _, row in df.iterrows(): collections[row["parent"]][row["predicted_label"]].append(row["table"])](http://vscodecontentref/27).
# Example: collections["Orders"]["embed"] = ["Lineitem"].
# Why This Step? Structures predictions into schema format. defaultdict avoids KeyError.
# Output: Basic schema dict (e.g., {"Orders": {"embed": ["Lineitem"], "reference": []}}).
# Step 3: Resolve Conflicts

# What Happens: Fixes issues like a table embedded in multiple parents (e.g., Lineitem only in Orders).
# How It Works (Code Breakdown):
# resolve_embedding_conflicts(): Uses embedding_priority (e.g., Lineitem → Orders only). Moves extras to "reference".
# Why This Step? Prevents data duplication/errors in NoSQL.
# Output: Cleaned collections.
# Step 4: Apply Domain Rules

# What Happens: Overrides with expert knowledge (e.g., always embed Lineitem in Orders).
# How It Works (Code Breakdown):
# apply_tpch_rules(): Forces embeds (e.g., collections["Orders"]["embed"].append("Lineitem")), embeds small dims.
# Calls resolve_embedding_conflicts() again.
# Why This Step? ML isn't perfect; rules ensure best practices (e.g., performance for TPC-H).
# Output: Refined schema.
# Step 5: Add Standalone Tables

# What Happens: Includes tables with no relationships.
# How It Works (Code Breakdown):
# Loop: for table in tpch_query_usage.keys(): if table not in collections: collections[table] = {"embed": [], "reference": []}.
# Why This Step? Ensures complete schema.
# Output: Full collections dict.

# Step 6: Validate Schema

# What Happens: Checks for errors (cycles, duplicates, conflicts).
# How It Works (Code Breakdown):
# validate_schema(): Uses NetworkX to check DAG (no cycles), counts embeddings, checks conflicts.
# Prints warnings if invalid.
# Why This Step? Catches bad schemas (e.g., infinite embeds).
# Output: Boolean (valid/invalid) and prints.
# Step 7: Visualize and Analyze

# What Happens: Draws graph and computes stats.
# How It Works (Code Breakdown):
# visualize_schema(): Uses NetworkX/Matplotlib to draw nodes (red= facts, blue=dims), edges (green=embed, blue=reference).
# analyze_schema(): Counts embeds/references, calculates ratio/depth.
# Why This Step? Makes schema understandable; stats show balance (e.g., 83% embed ratio).
# Output: Graph image and stats (e.g., "Total embeddings: 5").
# Full Schema Generation Flow Summary: Predictions → Build Collections → Resolve Conflicts → Apply Rules → Add Tables → Validate → Visualize/Analyze. Result: A complete, validated NoSQL schema.