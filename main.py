import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load your data
df = pd.read_csv("retaildata.csv")

# Drop the Trans_ID column, we only care about the products
df = df.drop('Trans_ID', axis=1)

# Convert each row into a list of products (drop NaNs)
transactions = df.apply(lambda row: row.dropna().tolist(), axis=1).tolist()

# Use TransactionEncoder to one-hot encode the data
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
te_df = pd.DataFrame(te_array, columns=te.columns_)

# Get frequent itemsets
frequent_itemsets = apriori(te_df, min_support=0.2, use_colnames=True)

# Get association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Show results
print("\n✅ Frequent Itemsets:\n", frequent_itemsets)
print("\n🔗 Association Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])




# Sort itemsets by support
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

# Keep only top N
top_itemsets = frequent_itemsets.head(10)

# Convert frozensets to strings for labeling
top_itemsets['itemsets'] = top_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))

# Plot
plt.figure(figsize=(10,6))
plt.barh(top_itemsets['itemsets'], top_itemsets['support'], color='skyblue')
plt.xlabel('Support')
plt.title('Top Frequent Itemsets')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

