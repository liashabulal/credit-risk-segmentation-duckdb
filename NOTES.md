# NOTES

## Optimization ideas

### `credit_limit` column uses a slow Python for-loop instead of vectorized numpy

**What the issue is:**
The `credit_limit` column is built with a Python list comprehension that loops over all 500,000 customers one at a time, calling `np.random.choice(...)` separately for each one:

```python
'credit_limit': np.array([
    np.random.choice(
        [5000, 10000, 25000, 50000],
        p=[0.6, 0.3, 0.08, 0.02] if r > 0.4 else [0.1, 0.2, 0.4, 0.3]
    ) for r in customer_risk
]),
```

**Why it matters / why it happened:**
Every other column in this dataset (age, months_on_book, etc.) is generated in one shot using a single numpy call that works on the whole array at once ("vectorized"). This column instead loops through the data row-by-row in plain Python, and on top of that, it calls a fairly heavy function (`np.random.choice` with probability weights) 500,000 separate times. Each call has its own setup overhead, so the cost adds up fast. It likely happened because the probabilities depend on whether a customer's risk score is above or below 0.4, and writing an if/else per-row in a loop felt like the simplest way to express that condition — vectorizing conditional logic is less obvious than vectorizing a plain distribution draw.

**What the best fix would be:**
Split the customers into two groups based on the 0.4 risk threshold, generate credit limits for each group in one vectorized `np.random.choice` call (using the full group size as the `size` argument), and then combine the results back together in the right order. This does the exact same random assignment, just in two big batches instead of 500,000 tiny ones — same output, far less time.

---

### Transactions table builds `customer_risks` with a Python loop over 5,000,000 rows (bigger version of the same problem)

**What the issue is:**
Right after the transactions are set up, the notebook does this to attach each customer's risk score onto every transaction:

```python
customer_risk_map = customers.set_index('customer_id')['risk_score'].to_dict()
cust_ids = np.random.randint(1, n+1, 5000000)
customer_risks = np.array([customer_risk_map[c] for c in cust_ids])
```

It first builds a Python dictionary with 500,000 entries, then loops through all 5,000,000 transaction rows one at a time, looking up each customer's risk score in that dictionary.

**Why it matters:**
This is the same "loop instead of vectorized" pattern as the `credit_limit` issue, except it runs 10x more times — 5 million loop iterations instead of 500,000. Each pass through the loop has to look something up in the dictionary and hand it back as a plain Python number, and doing that 5 million times adds up to real, noticeable slowdown. It's likely the single slowest line in the whole notebook.

**Why it happened:**
The transactions table is much bigger than the customers table, so each transaction needs to "look up" its customer's risk score from the smaller table. A dictionary lookup is a very natural way to think about "look up a value by ID," so it's an easy pattern to reach for even though it's not the fastest option here.

**What the best fix would be:**
Since `customer_id` values run consecutively from 1 to 500,000, there's no need for a dictionary at all — the position in the array already tells you which customer it belongs to. You can get the same result in one line with plain numpy indexing: `customers['risk_score'].values[cust_ids - 1]`. This does the exact same lookup for all 5 million rows at once instead of one at a time, and skips building the dictionary in the first place.

---

### Same model gets trained multiple times from scratch instead of being reused

**What the issue is:**
The notebook trains fresh copies of the same models more than once for no functional reason:
- `cross_val_score` (cell with the `models` dict) already trains 5 Logistic Regression models and 5 Random Forest models internally to compute the AUC scores, but throws those trained models away afterward.
- Later, a brand new `RandomForestClassifier` is created and fit again from scratch just to read off its feature importances for the dashboard chart.
- Even later, a new `LogisticRegression` pipeline is created and fit again from scratch just to generate the final approve/review/decline predictions.

**Why it matters:**
Training a Random Forest with 100 trees on 500,000 rows, or a Logistic Regression on the same data, isn't instant — doing it two or three separate times when once would do wastes time for no benefit. It also creates a subtle risk: if someone later tweaks the model settings in one spot but forgets to update the other spot, the chart and the final predictions could quietly stop matching what was actually cross-validated.

**Why it happened:**
`cross_val_score` is convenient for quickly getting a score, but by design it doesn't hand back the trained models — so as soon as you need the trained model itself for something else (like feature importances or predictions), it feels natural to just fit a new one right there instead of restructuring the earlier step.

**What the best fix would be:**
Use `cross_validate(..., return_estimator=True)` instead of `cross_val_score` so the fitted models from cross-validation are kept, or simply fit each model once in a single place and reuse that same fitted model everywhere it's needed (scoring, feature importances, and final predictions) instead of re-fitting it fresh each time.

---

### Minor: the `features` list is defined twice with the same values

**What the issue is:**
`features = ['utilization_rate', 'months_on_book', 'credit_limit', 'raw_risk_score']` is written once when setting up the models, and then written again, identically, later on when building the feature importance chart.

**Why it matters:**
This isn't a speed problem, just a small maintenance risk — if someone updates the feature list in one place (say, to add a new column) but forgets the second copy exists, the chart could end up showing importances for a different set of features than the one actually used to train the models.

**What the best fix would be:**
Define `features` once, near the top of the notebook, and reuse that same variable everywhere instead of retyping the list a second time.

---

## Interview talking points

### `credit_limit` column: loops vs. vectorization

**What the issue is:**
This is a classic "loop instead of vectorized operation" performance problem — a very common thing interviewers ask about when they want to see if someone understands why numpy/pandas code is fast.

**Why it matters / why it happened:**
Plain Python loops are slow because each iteration has interpreter overhead — Python has to figure out types, manage memory, and do bookkeeping every single time through the loop. Numpy avoids this by doing the same operation on an entire array at once, in fast compiled C code, only paying that overhead once instead of 500,000 times. The for-loop version here happened because the logic needed an if/else condition per row, which is easy to write as a loop but harder to immediately see how to vectorize.

**What the best fix would be:**
Explain that instead of looping, you can split the data into groups based on the condition (e.g., risk > 0.4 vs. not) and apply the vectorized operation to each whole group at once. This is a good story to tell in an interview: it shows you can spot a performance bottleneck, explain in plain terms why it's slow (Python loop overhead vs. compiled batch operations), and describe a concrete fix rather than just saying "vectorize it."

---

### `customer_risks` mapping: dictionary lookup in a loop vs. array indexing

**What the issue is:**
This is the same category of problem as the `credit_limit` loop, but scaled up — a plain Python loop with a dictionary lookup running 5,000,000 times instead of 500,000. It's a great example to bring up because it shows the cost isn't just "loops are slow," it's "loops get proportionally worse the bigger your data gets."

**Why it matters:**
A dictionary lookup inside a Python loop has to hash the key, look it up, and hand back a Python object every single time — none of that happens in bulk, unlike a numpy array operation. When the ID values are just consecutive integers, that whole lookup step is unnecessary, because the position in the array already encodes the identity you're looking up.

**What the best fix would be:**
Talk through recognizing when a lookup can be replaced with direct indexing — here, `customer_id` running from 1 to n means `values[cust_ids - 1]` gives the exact same answer as the dictionary, with no loop and no hashing at all. It's a good example of the broader interview point: before optimizing a loop, first ask if the loop is even necessary.

---

### Re-fitting the same model multiple times

**What the issue is:**
The notebook fits fresh Random Forest and Logistic Regression models more than once — once inside cross-validation, and again later just to get feature importances or final predictions — instead of reusing a trained model.

**Why it matters:**
This is a good talking point about being efficient with expensive operations, not just efficient with loops: training a model is one of the more computationally costly steps in any ML pipeline, so doing it redundantly is a bigger waste than most inefficient one-liners elsewhere in the code. It also shows you think about maintainability — duplicated training code can quietly drift out of sync if only one copy gets updated.

**What the best fix would be:**
Mention `cross_validate(..., return_estimator=True)` as the built-in scikit-learn way to keep trained models from cross-validation, or simply structuring the notebook so each model is trained once and its fitted object is reused wherever it's needed downstream.
