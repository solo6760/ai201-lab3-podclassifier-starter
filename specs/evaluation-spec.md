# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
Accuracy = (Number of correct predictions) / (Total number of predictions)
Where a prediction is "correct" if it exactly matches the ground-truth label for that episode.
```

---

**Step-by-step logic:**

```
1. Verify if the predictions list is empty (length == 0). If so, return 0.0.
2. Initialize a counter for correct predictions to 0.
3. Loop through both predictions and ground_truth lists simultaneously using their indexes (or zip).
4. For each index, compare predictions[i] with ground_truth[i]. If they match exactly, increment the correct predictions counter.
5. Divide the correct predictions count by the total number of predictions, and return this float value.
```

---

**Edge case — what if both lists are empty?**

```
Return 0.0. If there are no predictions or ground truth labels, accuracy is technically undefined, and returning 0.0 prevents a division by zero error.
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

1. Length of lists is 4 (not empty).
2. Comparisons:
   - Index 0: "interview" == "interview" -> Correct (1)
   - Index 1: "solo" == "solo" -> Correct (2)
   - Index 2: "panel" == "solo" -> Incorrect
   - Index 3: "interview" == "narrative" -> Incorrect
3. Total correct = 2. Total predictions = 4.
4. Return 2 / 4 = 0.5.
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
An episode counts as correctly classified for a class (e.g., "interview") if its ground_truth label is "interview" and its predicted label is also "interview".
```

---

**What does "total" mean for a given class?**

```
"Total" is the total number of episodes in the dataset that have that specific ground_truth label. It is not the total number of predictions made for that label.
```

---

**Step-by-step logic:**

```
1. Initialize a dict where each label in VALID_LABELS maps to {"correct": 0, "total": 0, "accuracy": 0.0}.
2. Loop over predictions and ground_truth lists simultaneously (using zip).
3. For each pair (pred, truth):
   a. If truth is in VALID_LABELS:
      - Increment the "total" count for that truth class in our dict.
      - If pred == truth, increment the "correct" count for that truth class in our dict.
4. After the loop, iterate through each label in VALID_LABELS:
   a. If total is greater than 0, set accuracy to correct / total.
   b. If total is 0, set accuracy to 0.0.
5. Return the completed dict.
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
The accuracy should be set to 0.0, and the total/correct counts should be 0. This avoids a ZeroDivisionError and indicates that we have no data points to assess performance for this class.
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

label       correct  total  accuracy
----------  -------  -----  --------
interview   1        1      1.0
solo        1        2      0.5
panel       1        1      1.0
narrative   0        1      0.0
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?
