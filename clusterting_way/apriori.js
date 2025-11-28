// Step 1: Hardcoded transactional data
const transactions = [
    new Set(['milk', 'bread', 'butter']),
    new Set(['bread', 'biscuit']),
    new Set(['milk', 'bread', 'biscuit']),
    new Set(['bread', 'butter', 'namkeen']),
    new Set(['milk', 'bread', 'butter', 'egg']),
];

// Step 2: Minimum support threshold (e.g., 40%)
const minSupport = 0.4;

// Step 3: Function to calculate support of an itemset
function calculateSupport(itemset, transactions) {
    let count = 0;
    for (const transaction of transactions) {
        if ([...itemset].every(item => transaction.has(item))) {
            count++;
        }
    }
    return count / transactions.length;
}

// Step 4: Generate frequent 1-itemsets
function findFrequentItemsets(transactions, minSupport) {
    const items = new Set();
    transactions.forEach(transaction => {
        transaction.forEach(item => items.add(item));
    });

    // Generate 1-itemsets and calculate support
    const frequentItemsets = new Map();
    for (const item of items) {
        const itemset = new Set([item]);
        const support = calculateSupport(itemset, transactions);
        if (support >= minSupport) {
            frequentItemsets.set(itemset, support);
        }
    }

    return frequentItemsets;
}

// Step 5: Generate larger frequent itemsets iteratively
function generateFrequentItemsets(frequentItemsets, transactions, minSupport) {
    const allFrequentItemsets = new Map([...frequentItemsets]);
    let k = 2;

    while (true) {
        const candidates = [];
        const itemsets = [...frequentItemsets.keys()];

        // Generate candidate k-itemsets
        for (let i = 0; i < itemsets.length; i++) {
            for (let j = i + 1; j < itemsets.length; j++) {
                const unionSet = new Set([...itemsets[i], ...itemsets[j]]);
                if (unionSet.size === k) {
                    candidates.push(unionSet);
                }
            }
        }

        // Calculate support and filter candidates
        const currentFrequentItemsets = new Map();
        for (const candidate of candidates) {
            const support = calculateSupport(candidate, transactions);
            if (support >= minSupport) {
                currentFrequentItemsets.set(candidate, support);
            }
        }

        if (currentFrequentItemsets.size === 0) break; // No more frequent itemsets

        // Add to all frequent itemsets
        for (const [itemset, support] of currentFrequentItemsets) {
            allFrequentItemsets.set(itemset, support);
        }

        frequentItemsets = currentFrequentItemsets; // Update for next iteration
        k++;
    }

    return allFrequentItemsets;
}

// Main Function
const frequent1Itemsets = findFrequentItemsets(transactions, minSupport);
const allFrequentItemsets = generateFrequentItemsets(frequent1Itemsets, transactions, minSupport);

// Display Results
console.log("Final Frequent Itemsets:");
for (const [itemset, support] of allFrequentItemsets) {
    console.log(`${[...itemset].join(', ')}: ${support}`);
}
