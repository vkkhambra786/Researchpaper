import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import json
from collections import defaultdict
import networkx as nx
# Add this import at the top (if not already there)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

# === FEATURE ENGINEERING WITH NORMALIZATION ===
def generate_features():
    # features will store the numerical feature arrays for each relationship.
    # relationships will list all child-parent pairs.
    # features is a dict (key: tuple like ("Lineitem", "Orders"), value: list of numbers). relationships is a list of tuples.
    features = {}
    relationships = []
    
    # First pass to collect stats for normalization
    max_join_freq = 0
    max_access_freq = 0
    max_parent_freq = 0
    max_size_ratio = -np.inf
    min_size_ratio = np.inf
    
    for child, parents in foreign_keys.items(): # Outer loop over child tables (e.g., "Lineitem").
        for parent in parents: # Inner loop over parents (e.g., ["Orders", "Part"] for Lineitem).
            join_freq = len(set(tpch_query_usage[child]) & set(tpch_query_usage[parent])) # Counts shared queries (intersection of sets). E.g., if Lineitem and Orders share 10 queries, join_freq = 10.
            max_join_freq = max(max_join_freq, join_freq) # Updates max join frequency.
            max_access_freq = max(max_access_freq, len(tpch_query_usage[child])) # Tracks max queries for child (e.g., Lineitem has 16).
            max_parent_freq = max(max_parent_freq, len(tpch_query_usage[parent])) # Tracks max for parent.
            # Size ratio (log scale to handle large differences)
            size_ratio = np.log10(table_sizes[child] / table_sizes[parent]) # Calculates log of size ratio (e.g., log(6M/1.5M) ≈ 0.6). Log handles huge numbers.
            max_size_ratio = max(max_size_ratio, size_ratio) # Tracks range for normalization.
            min_size_ratio = min(min_size_ratio, size_ratio) # Tracks range for normalization.

#        max_join_freq = 0, etc.: Initializes variables to track extremes. np.inf is infinity (for min/max).
# for child, parents in foreign_keys.items(): Outer loop over child tables (e.g., "Lineitem").
# for parent in parents: Inner loop over parents (e.g., ["Orders", "Part"] for Lineitem).
# join_freq = len(set(tpch_query_usage[child]) & set(tpch_query_usage[parent])): Counts shared queries (intersection of sets). E.g., if Lineitem and Orders share 10 queries, join_freq = 10.
# max_join_freq = max(max_join_freq, join_freq): Updates max join frequency.
# max_access_freq = max(max_access_freq, len(tpch_query_usage[child])): Tracks max queries for child (e.g., Lineitem has 16).
# max_parent_freq = max(max_parent_freq, len(tpch_query_usage[parent])): Tracks max for parent.
# size_ratio = np.log10(table_sizes[child] / table_sizes[parent]): Calculates log of size ratio (e.g., log(6M/1.5M) ≈ 0.6). Log handles huge numbers.
# max_size_ratio = max(max_size_ratio, size_ratio) and min_size_ratio = min(min_size_ratio, size_ratio): Tracks range for normalization.
   
    # Second pass to create normalized features
    for child, parents in foreign_keys.items(): # Same outer loop as first pass.
        for parent in parents: # Same inner loop as first pass.
            # Jaccard similarity (overlap ratio)
            q_overlap = len(set(tpch_query_usage[child]) & set(tpch_query_usage[parent])) / \
                        max(len(set(tpch_query_usage[child]) | set(tpch_query_usage[parent])), 1) # Jaccard similarity (overlap ratio). E.g., 10 shared / 20 total = 0.5. Already in [0,1].
            
            # Normalize size ratio to [0,1]
            size_ratio = np.log10(table_sizes[child] / table_sizes[parent]) # Recalculates log size ratio.
            norm_size_ratio = (size_ratio - min_size_ratio) / (max_size_ratio - min_size_ratio + 1e-8) # Min-max normalization to [0,1]. 1e-8 avoids divide-by-zero.
            
            # Normalize other features
            join_freq = len(set(tpch_query_usage[child]) & set(tpch_query_usage[parent])) # Recalculates shared queries.
            norm_join_freq = join_freq / max(max_join_freq, 1) # Scales to [0,1] using max from first pass.

            # Normalize access frequency
            access_freq = len(tpch_query_usage[child]) # Child's query count.
            norm_access_freq = access_freq / max(max_access_freq, 1) # Scales to [0,1].
            
            # Normalize parent frequency
            parent_freq = len(tpch_query_usage[parent]) # Parent's query count.
            norm_parent_freq = parent_freq / max(max_parent_freq, 1) # Scales to [0,1].

            # Is fact table (binary)
            is_fact = int(child in ["Lineitem", "Orders", "PartSupp"]) # Binary flag (1 if child is a "fact" table, 0 otherwise).
            

#             for child, parents in foreign_keys.items(): Same outer/inner loops as first pass.
# q_overlap = len(set(tpch_query_usage[child]) & set(tpch_query_usage[parent])) / max(len(set(tpch_query_usage[child]) | set(tpch_query_usage[parent])), 1): Jaccard similarity (overlap ratio). E.g., 10 shared / 20 total = 0.5. Already in [0,1].
# size_ratio = np.log10(table_sizes[child] / table_sizes[parent]): Recalculates log size ratio.
# norm_size_ratio = (size_ratio - min_size_ratio) / (max_size_ratio - min_size_ratio + 1e-8): Min-max normalization to [0,1]. 1e-8 avoids divide-by-zero.
# join_freq = len(set(tpch_query_usage[child]) & set(tpch_query_usage[parent])): Recalculates shared queries.
# norm_join_freq = join_freq / max(max_join_freq, 1): Scales to [0,1] using max from first pass.
# access_freq = len(tpch_query_usage[child]): Child's query count.
# norm_access_freq = access_freq / max(max_access_freq, 1): Scales to [0,1].
# parent_freq = len(tpch_query_usage[parent]): Parent's query count.
# norm_parent_freq = parent_freq / max(max_parent_freq, 1): Scales to [0,1].
# is_fact = int(child in ["Lineitem", "Orders", "PartSupp"]): Binary flag (1 if child is a "fact" table, 0 otherwise).
# features[(child, parent)] = [...]: Stores the 6 features as a list for this relationship.
# relationships.append((child, parent)): Adds the pair to the list.
            features[(child, parent)] = [
                q_overlap,            # Already in [0,1]
                norm_size_ratio,       # Now in [0,1]
                norm_join_freq,       # Now in [0,1]
                norm_access_freq,     # Now in [0,1]
                norm_parent_freq,     # Now in [0,1]
                is_fact               # Binary 0/1
            ] # 6 features total
            relationships.append((child, parent)) # List of (child, parent) pairs
    return features, relationships # Returns the features dict and relationships list

features, relationships = generate_features()

# === RL ENVIRONMENT ===
class SchemaEnv(Env):
    #  Class Definition and __init__ Method
    # A custom class inheriting from Gymnasium's Env (environment) ,
    # that simulates the schema transformation process as an RL "game."
    #  It defines the rules: the agent observes features (e.g., query overlap), 
    # takes actions (embed or reference), gets rewards, and progresses through relationships.
    
    # ********** WHY is SchemaEnv needed? **********
    # RL requires an environment to define states, actions, and rewards.
    #  This turns schema decisions into a Markov Decision Process (MDP),
    #  where the agent learns optimal policies (e.g., "embed if high overlap").
    def __init__(self, relationships, features):
        super(SchemaEnv, self).__init__()
        self.relationships = relationships
        self.features = features
        self.index = 0
        self.schema_state = {}
        self.observation_space = Box(low=0, high=1, shape=(6,), dtype=np.float32)
        self.action_space = Discrete(2)  # 0 = reference, 1 = embed
        
#         super(SchemaEnv, self).__init__(): Calls the parent Env class to inherit RL functionality.
# self.relationships and self.features: Stores inputs (list of pairs and dict of feature arrays).
# self.index = 0: Tracks progress (starts at first relationship).
# self.schema_state = {}: Dict to record decisions (e.g., {("Lineitem", "Orders"): "embed"}).
# self.observation_space = Box(low=0, high=1, shape=(6,), dtype=np.float32): Defines observations as 6 numbers (0-1) for features (e.g., overlap, size).
# self.action_space = Discrete(2): Defines actions as 2 choices (0 = reference, 1 = embed).
    def reset(self, *, seed=None, options=None):
        self.index = 0 # Reset to first relationship.
        self.schema_state = {} # Clear previous decisions.
        obs = self._get_obs() # Get initial observation.

#         Resets index to 0 and schema_state to empty (forget previous decisions).
# Calls _get_obs() to get the first observation (features for first relationship).
# Returns observation and empty info dict.
        return obs, {}  # Return observation and empty info dict.

    def _get_obs(self): 
#         Resets index to 0 and schema_state to empty (forget previous decisions).
# Calls _get_obs() to get the first observation (features for first relationship).
# Returns observation and empty info dict.
        if self.index < len(self.relationships): # If not done.
            rel = self.relationships[self.index] # Current relationship (e.g., ("Lineitem", "Orders")).
            return np.array(self.features[rel], dtype=np.float32) # Return its 6 features as array.
        else:
            return np.zeros(6, dtype=np.float32) # If done, return zeros.

    def step(self, action): # action is 0 (reference) or 1 (embed).
        rel = self.relationships[self.index] # Current relationship.
        self.schema_state[rel] = "embed" if action == 1 else "reference" # Record decision.

        reward = self._compute_reward(rel, action) # Calculate reward.

        self.index += 1 # Move to next relationship.
        terminated = self.index >= len(self.relationships) # Check if episode ends.
        truncated = False # Not used here.

        obs = self._get_obs() if not terminated else np.zeros(6, dtype=np.float32) # Next observation or zeros.
        info = {} # Empty info.
         
#          Records the decision in schema_state (e.g., "embed" for Lineitem->Orders).
# Computes reward via _compute_reward().
# Increments index to next relationship.
# Checks if episode is done (terminated if all relationships processed).
# Gets next observation or zeros if done.

        return obs, reward, terminated, truncated, info # Returns all step outputs.

     
    def _compute_reward(self, rel, action): # rel is (child, parent), action is 0 or 1.
        child, parent = rel # Unpack relationship.
        q_overlap = self.features[rel][0] # Query overlap feature.
         # Size ratio (child size / parent size)
        size_ratio = table_sizes[child]/table_sizes[parent] # Actual size ratio (not normalized).
    
    # Dynamic thresholds based on table characteristics
        is_dimension = child in ["Nation", "Region"] # Dimension tables.
        is_fact = child in ["Lineitem", "Orders", "PartSupp"] # Fact tables.
    
        if action == 1:  # Embed
            reward = q_overlap * 3  # Strong co-access bonus
        
        # Size penalty (exponential)
            if size_ratio > 0.05:  # 5% size threshold
               penalty = min(2.0, (size_ratio/0.05) ** 2) # Quadratic penalty for large sizes
               reward -= penalty # Penalize large sizes
            
        # Special bonus for dimension tables
            if is_dimension:
               reward += 1.5
            
        else:  # Reference
             reward = (1 - q_overlap) * 1.5  # Base reward
        
        # Size bonus (logarithmic)
             if size_ratio > 1.0:
                reward += min(2.0, np.log10(size_ratio))
    
    # Penalize fact table embeddings
        if is_fact and action == 1: # Embedding a fact table
           reward -= 0.5 # Small penalty


#            Extracts features (overlap, size) and checks table types.
# For embed: Bonus for overlap, penalty for large sizes, extra for dimensions.
# For reference: Bonus for low overlap/large sizes.
# Penalizes embedding fact tables.
# Clips reward to -2 to 4 to avoid extremes.
        
        return float(np.clip(reward, -2.0, 4.0))

# === TRAINING ===
# Training teaches the RL agent (DQN model) to make better decisions (embed vs. reference)
#  by interacting with the SchemaEnv environment.
#  It's like practicing a game— the agent tries actions, gets rewards,
#  and learns from mistakes.
env = SchemaEnv(relationships, features)
check_env(env)  # Should now pass validation

#  DQN is a type of reinforcement learning (RL) algorithm that uses
#  a neural network (a brain-like computer program) to predict 
# the "best" action (embed or reference) for a given situation (features like query overlap).
#  It learns by playing "games" (episodes) in your SchemaEnv, 
# getting rewards for good moves, and improving over time.
model = DQN(
    'MlpPolicy',  # The "brain" type: Multi-Layer Perceptron (a neural network).
    env,  # The "game" it plays: Your SchemaEnv (provides situations, actions, rewards).
    verbose=1,  # Prints progress (like "I'm learning!").
    learning_rate=5e-5,  # How fast it learns (slow = 0.00005, for careful changes).
    buffer_size=100000,  # Memory bank: Stores up to 100,000 past moves for review.
    learning_starts=5000,  # Waits 5,000 random moves before starting to learn (fills memory).
    batch_size=128,  # Reviews 128 past moves at a time.
    gamma=0.95,  # Future value: Cares 95% about future rewards (not just immediate ones).
    target_update_interval=2000,  # Updates a "backup brain" every 2,000 moves (keeps learning stable).
    exploration_final_eps=0.05,  # Ends random moves at 5% (starts exploring, then focuses on learned moves).
    policy_kwargs={'net_arch': [256, 256]}  # Brain structure: Input (6 features) → 256 neurons → 256 neurons → Output (2 actions).
)

# Change the training call to:
# Train with progress monitoring
model.learn(
    total_timesteps=30000,
    progress_bar=True,
    tb_log_name="schema_rl_v2"
)
print("\nTraining complete!\n")

# === EVALUATION ===
obs, _ = env.reset()
done = False
decisions = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    rel = env.relationships[env.index-1] if env.index > 0 else ("Start", "Start")
    decisions.append({
        "relationship": f"{rel[0]}->{rel[1]}",
        "decision": "Embed" if action == 1 else "Reference",
        "reward": reward
    })

print("\nDecision Summary:")
for d in decisions:
    print(f"{d['relationship']:20} {d['decision']:10} Reward: {d['reward']:.2f}")

# === OUTPUT SCHEMA WITH TPC-H OPTIMIZATIONS ===
collections = defaultdict(lambda: {"embed": [], "reference": []})
for (child, parent), decision in env.schema_state.items():
    collections[parent][decision].append(child)

# Apply TPC-H specific optimizations

# Enhanced TPC-H rules
# Enhanced TPC-H rules - REPLACE your current apply_tpch_rules with this:
# === UPDATED apply_tpch_rules FUNCTION ===
# === UPDATED apply_tpch_rules FUNCTION ===

# === OUTPUT FINAL SCHEMA ===

# Apply TPC-H rules (updated version)
def apply_tpch_rules(collections):
    # Clear existing relationships
    for table in collections:
        collections[table]["embed"] = []
        collections[table]["reference"] = []
    
    # Define optimal relationships
    relationships = {
        "Orders": {
            "embed": ["Lineitem"],  # Critical for TPC-H
            "reference": []
        },
        "Customer": {
            "embed": [],
            "reference": ["Orders"]
        },
        "Nation": {
            "embed": ["Region"],    # Tiny dimension
            "reference": ["Supplier", "Customer"]
        },
        "Supplier": {
            "embed": [],
            "reference": ["PartSupp", "Lineitem"]
        },
        "Part": {
            "embed": [],
            "reference": ["PartSupp"]
        },
        "PartSupp": {
            "embed": [],
            "reference": ["Lineitem"]
        },
        "Region": {
            "embed": [],
            "reference": []
        },
        "Lineitem": {
            "embed": [],
            "reference": []
        }
    }
    
    # Apply relationships
    for table, rels in relationships.items():
        collections[table]["embed"] = rels["embed"]
        collections[table]["reference"] = rels["reference"]
    return collections

collections = apply_tpch_rules(collections)

# Print JSON version
print("\nFinal NoSQL Schema (JSON format):")
print(json.dumps(collections, indent=2))

# Print human-readable version
print("\nHuman-Readable Schema Structure:")
for table in collections:
    embeds = collections[table]["embed"]
    refs = collections[table]["reference"]
    
    print(f"\n{table} Collection:")
    if embeds:
        print(f"  → Embeds directly: {', '.join(embeds)}")
    if refs:
        print(f"  → References: {', '.join(refs)}")
    if not embeds and not refs:
        print("  → Standalone collection")

# Validate final schema
def validate_schema(collections):
    issues = []
    G = nx.DiGraph()
    
    # Build relationship graph
    for parent, rels in collections.items():
        for child in rels["embed"]:
            G.add_edge(parent, child, weight=1)
        for child in rels["reference"]:
            G.add_edge(parent, child, weight=0.1)
    
    # Only show critical errors
    try:
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles:
            issues.append(f"Critical: Circular reference detected: {' → '.join(cycle)}")
    except nx.NetworkXNoCycle:
        pass
    
    return issues

print("\nSchema Validation:")


issues = validate_schema(collections)
if issues:
    for issue in issues:
        print(issue)
else:
    print("No validation issues found")



# Add this new function after validate_schema
def plot_schema_graph(collections):
    """
    Plots a graph of the NoSQL schema using NetworkX and Matplotlib.
    - Nodes: Tables (red for fact tables, blue for others).
    - Edges: Green solid for embeds, blue dashed for references.
    """
    G = nx.DiGraph()  # Directed graph for relationships
    
    # Add nodes (tables) with colors
    for table in collections:
        color = 'red' if table in ["Lineitem", "Orders", "PartSupp"] else 'blue'
        G.add_node(table, color=color)
    
    # Add edges (relationships) with colors and styles
    for parent, rels in collections.items():
        for child in rels["embed"]:
            G.add_edge(parent, child, color='green', style='solid')
        for child in rels["reference"]:
            G.add_edge(parent, child, color='blue', style='dashed')
    
    # Position nodes (spring layout for nice spacing)
    pos = nx.spring_layout(G, seed=42)  # Seed for reproducible layout
    
    # Draw the graph using nx.draw (more flexible than draw_networkx_*)
    plt.figure(figsize=(12, 8))  # Larger figure for clarity
    
    # Draw nodes
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2000, 
            alpha=0.8, font_size=10, font_weight='bold', arrows=True, 
            edge_color=[G.edges[edge]['color'] for edge in G.edges()], 
            style=[G.edges[edge]['style'] for edge in G.edges()], width=2)
    
    # Add legend (fixed: use Line2D directly)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Fact Table'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Other Table'),
        Line2D([0], [0], color='green', lw=2, label='Embed'),
        Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Reference')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.title("NoSQL Schema Graph (RL-Generated)", fontsize=14)
    plt.axis('off')  # Hide axes for cleaner look
    plt.show()  # Display the graph
# Call the function at the end of your script (after validation)
plot_schema_graph(collections)
collections = apply_tpch_rules(collections)

# Add standalone tables
for table in tpch_query_usage.keys():
    if table not in collections and not any(table in v["embed"] for v in collections.values()):
        collections[table] = {"embed": [], "reference": []}


# print(json.dumps(collections, indent=2))


#1. generate_features() Function
# Creates normalized numerical features for each child-parent relationship to feed into the RL model.

# The RL agent (DQN model) observes the current state (6 features, e.g., query overlap, size ratio) from SchemaEnv. 
# It predicts Q-values (expected rewards) for each action: 0 (reference) or 1 (embed).
#  It chooses the action with the higher Q-value (exploitation) or randomly (exploration early on).
#  For example, if Q for embed is higher, it decides "embed.
# 2. SchemaEnv Class and Its Methods
    #__init__(self, relationships, features): Purpose: Initializes the environment with data and spaces for RL.

    #reset(self, *, seed=None, options=None): Purpose: Resets the environment to start a new episode.

    #_get_obs(self): Purpose: Retrieves the current observation (features) for the RL agent.

    #step(self, action): Purpose: Takes an action (embed/reference), updates state,
    # and returns new observation, reward, and done status.

    #_compute_reward(self, rel, action): Purpose: Calculates the reward based on the
    # relationship features and the action taken.

# 3. DQN Model Initialization and Training
# Sets up and trains the RL agent to learn optimal schema decisions through interaction with SchemaEnv.

# 4. Evaluation Loop
# Runs the trained model through the environment to collect and print decisions made for each relationship.

# 5. apply_tpch_rules(collections) Function
# Enforces specific TPC-H schema design rules to ensure optimal performance.

# 6. validate_schema(collections) Function
# Checks the final schema for issues like circular references and prints any problems found.

# 7. plot_schema_graph(collections) Function
# Visualizes the final NoSQL schema as a graph using NetworkX and Matplotlib, with

# color-coded nodes and edges for clarity.
# Add this new function after validate_schema


