# Step 1: Dummy transactional data (transactions contain items)
transactions = [
    {'milk', 'bread', 'butter'},
    {'bread', 'biscuit'},
    {'milk', 'bread', 'biscuit'},
    {'bread', 'butter', 'namkeen'},
    {'milk', 'bread', 'butter', 'egg'}
]

# Step 2: Minimum support threshold (e.g., 40%)
min_support = 0.4

# Step 3: Function to calculate support of an itemset
def calculate_support(itemset, transactions):
    count = sum(1 for transaction in transactions if itemset.issubset(transaction))
    return count / len(transactions)

# Step 4: Generate frequent 1-itemsets
items = {item for transaction in transactions for item in transaction}  # Unique items
itemsets = [{item} for item in items]  # Convert to single-item sets
frequent_itemsets = {}

print("Step 4: Finding Frequent 1-Itemsets")
for itemset in itemsets:
    support = calculate_support(itemset, transactions)
    if support >= min_support:
        frequent_itemsets[frozenset(itemset)] = support
print(f"Frequent 1-Itemsets: {frequent_itemsets}\n")

# Step 5: Generate larger frequent itemsets iteratively
k = 2  # Start with pairs
while True:
    print(f"Step {k+3}: Generating Frequent {k}-Itemsets")
    # Generate candidate itemsets of size k
    candidates = [
        frozenset(i.union(j))
        for i in frequent_itemsets.keys()
        for j in frequent_itemsets.keys()
        if len(i.union(j)) == k
    ]
    # Remove duplicates
    candidates = list(set(candidates))
    
    # Calculate support and filter candidates
    current_frequent_itemsets = {}
    for candidate in candidates:
        support = calculate_support(candidate, transactions)
        if support >= min_support:
            current_frequent_itemsets[candidate] = support
    
    if not current_frequent_itemsets:
        break  # No more frequent itemsets
    frequent_itemsets.update(current_frequent_itemsets)
    print(f"Frequent {k}-Itemsets: {current_frequent_itemsets}\n")
    k += 1

# Final Output: All frequent itemsets
print("Final Frequent Itemsets:")
for itemset, support in frequent_itemsets.items():
    print(f"{set(itemset)}: {support}")
