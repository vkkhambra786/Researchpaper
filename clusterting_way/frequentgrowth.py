# Step 1: Hardcoded transactional data
transactions = [
    ['milk', 'bread', 'butter'],
    ['bread', 'biscuit'],
    ['milk', 'bread', 'biscuit'],
    ['bread', 'butter', 'namkeen'],
    ['milk', 'bread', 'butter', 'egg']
]

# Step 2: Minimum support threshold
min_support = 2  # Minimum number of occurrences

# Step 3: Count items and filter by minimum support
def get_frequent_items(transactions, min_support):
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + 1
    # Filter out items that do not meet the minimum support
    frequent_items = {item: count for item, count in item_counts.items() if count >= min_support}
    return frequent_items

frequent_items = get_frequent_items(transactions, min_support)
print("Frequent Items:", frequent_items)

# Step 4: Build the FP-Tree
class FPTreeNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None  # Link to the next node with the same item

    def increment(self, count):
        self.count += count

class FPTree:
    def __init__(self):
        self.root = FPTreeNode(None, 0, None)
        self.header_table = {}

    def build_tree(self, transactions, frequent_items):
        for transaction in transactions:
            # Sort items in transaction by frequency
            sorted_items = [item for item in transaction if item in frequent_items]
            sorted_items.sort(key=lambda item: frequent_items[item], reverse=True)
            self.insert_tree(sorted_items, self.root)

    def insert_tree(self, items, node):
        if not items:
            return
        first_item = items[0]
        if first_item in node.children:
            node.children[first_item].increment(1)
        else:
            new_node = FPTreeNode(first_item, 1, node)
            node.children[first_item] = new_node
            # Update header table
            if first_item in self.header_table:
                current = self.header_table[first_item]
                while current.link:
                    current = current.link
                current.link = new_node
            else:
                self.header_table[first_item] = new_node
        # Recursively insert remaining items
        self.insert_tree(items[1:], node.children[first_item])

# Build the FP-Tree
fp_tree = FPTree()
fp_tree.build_tree(transactions, frequent_items)

# Print the FP-Tree for visualization
def print_tree(node, indent=0):
    print('  ' * indent + f"{node.item}: {node.count}")
    for child in node.children.values():
        print_tree(child, indent + 1)

print("\nFP-Tree Structure:")
print_tree(fp_tree.root)

# Step 5: Extract frequent patterns from the FP-Tree
def mine_tree(fp_tree, min_support):
    patterns = {}

    # Iterate over items in the header table (bottom-up order)
    for item, node in sorted(fp_tree.header_table.items(), key=lambda x: x[0]):
        pattern_base = []
        # Follow links to gather all paths containing the item
        while node:
            path = []
            parent = node.parent
            while parent and parent.item:
                path.append(parent.item)
                parent = parent.parent
            if path:
                pattern_base.append((path, node.count))
            node = node.link
        # Count combinations of the pattern base
        pattern_counts = {}
        for path, count in pattern_base:
            for i in range(1, len(path) + 1):
                for subset in combinations(path, i):
                    subset = frozenset(subset)
                    pattern_counts[subset] = pattern_counts.get(subset, 0) + count
        # Filter patterns by minimum support
        for pattern, count in pattern_counts.items():
            if count >= min_support:
                patterns[frozenset(pattern | {item})] = count

    return patterns

from itertools import combinations
patterns = mine_tree(fp_tree, min_support)

# Output frequent patterns
print("\nFrequent Patterns:")
for pattern, count in patterns.items():
    print(f"{set(pattern)}: {count}")
