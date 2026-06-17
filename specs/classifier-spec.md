# Classifier Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 2.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `build_few_shot_prompt()` and
`classify_episode()` in `classifier.py`.

---

## build_few_shot_prompt(labeled_examples, description)

### What it does
Constructs a prompt string for the LLM that includes the task instructions,
all labeled training examples, and the new episode description to classify.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `labeled_examples` | `list[dict]` | Each dict has `"title"`, `"description"`, `"label"` (and others). These are the examples you labeled in Milestone 1. |
| `description` | `str` | The episode description to classify. |

### Output

| Return value | Type | Description |
|---|---|---|
| prompt | `str` | A complete prompt string ready to send to the LLM. |

---

### Spec fields — fill these in before writing code

**Task instruction (what should the LLM know about the task?):**

```
You are classifying podcast episodes by their format. Classify the episode
into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests,
  no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or
  discussing a topic together
- narrative: a story assembled from external sources — interviews, archival
  audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy.
```

---

**How should labeled examples be formatted in the prompt?**

```
Each example should include the episode title, a brief excerpt or the full
description, and the correct label. Separate examples with a blank line or
a delimiter like "---". Include all fields that help the model see why the
label was applied — title and description are both useful; other fields
(like episode ID) are not needed.
```

---

**Example block sketch (write one concrete example):**

```
Title: {title}
Description: {description}
Label: {label}
```

---

**How should the new episode (to be classified) be presented?**

```
Present it in the same format as the labeled examples, but omit the Label
line and replace it with an instruction to classify. For example:

Title: {title}
Description: {description}
Label: ?

Then add a line like: "Classify the episode above. Return your answer in
the format below:" followed by the output format you chose.
```

---

**What output format should you request from the LLM?**

```
Format:
LABEL: <label>
REASONING: <brief explanation>

We will request the following format:
LABEL: [interview | solo | panel | narrative]
REASONING: [1-2 sentences explaining the decision]

Tradeoffs considered:
1. Label on its own line: Simple, but fragile if the LLM includes prefix/conversational text.
2. JSON object: Standard but prone to parsing failures if the model includes markdown formatting (like ```json) or if the reasoning contains unescaped quotes/newlines.
3. LABEL: X / REASONING: Y (Key-value lines): Easy for the LLM to follow, and robust to parse using line-by-line checks or regex.
```

---

**Edge cases to handle in the prompt:**

```
1. Empty labeled_examples: If no examples are provided, the prompt will still provide the task instructions and definitions so that the model can perform zero-shot classification.
2. Short or empty description: The prompt will instruct the model to return "unknown" as the label if there is insufficient information to determine the format.
3. Whitespace & casing: Stripping whitespace and converting inputs to lowercase.
```

---

## classify_episode(description, labeled_examples)

### What it does
Classifies a single podcast episode description using the few-shot LLM classifier.
Returns a dict with a label and reasoning.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `description` | `str` | The episode description to classify. |
| `labeled_examples` | `list[dict]` | Labeled training examples from `load_labeled_examples()`. |

### Output

| Return value | Type | Description |
|---|---|---|
| result | `dict` | Must have keys `"label"` and `"reasoning"`. `"label"` must be one of `VALID_LABELS` or `"unknown"`. |

---

### Spec fields — fill these in before writing code

**Step 1 — Build the prompt:**

```
Call build_few_shot_prompt(labeled_examples, description) and store the
returned string in a variable (e.g., prompt). Pass through both arguments
exactly as received — no modification needed before calling.
```

---

**Step 2 — Send to the LLM:**

```
Call _client.chat.completions.create() with:
  - model: the model name from config (LLM_MODEL)
  - messages: a list with one dict — {"role": "user", "content": prompt}
    (system-design.md shows an optional system message too — either shape works)
  - max_tokens: a reasonable limit (e.g., 200–300) to keep responses concise

Extract the response text from:
  response.choices[0].message.content
```

---

**Step 3 — Parse the response:**

```
1. Split the raw response text into lines.
2. Initialize `label` as "unknown" and `reasoning` as the raw response text (or empty).
3. Search for a line starting with "LABEL:" (case-insensitive) and extract the label value after the colon, stripping whitespace and matching it against VALID_LABELS.
4. Search for a line starting with "REASONING:" (case-insensitive) and extract the text after it.
```

---

**Step 4 — Validate the label:**

```
1. Normalize the extracted label (strip whitespace, lowercase).
2. Check if it is in VALID_LABELS (interview, solo, panel, narrative).
3. If the label is not valid, default `label` to "unknown".
```

---

**Step 5 — Handle errors gracefully:**

```
Wrap the LLM call and parsing in a try-except block.
If any exception occurs (e.g., API connection error, rate limits, or unexpected response format):
  - Return {"label": "unknown", "reasoning": "Error: <exception details>"}
This prevents one failed API call from crashing the entire classification run.
```

---

### Return value structure

```python
{
    "label": str,      # one of VALID_LABELS, or "unknown" if invalid/error
    "reasoning": str,  # brief explanation from the LLM
}
```

---

## Notes on label quality

The classifier is only as good as your labels. If your training examples have
inconsistent or ambiguous labels, the LLM will learn the wrong pattern.

Before implementing the classifier, re-read `data/taxonomy.md` and double-check
any labels you're unsure about. Annotation quality is part of the lab.

---

## Implementation Notes

*Fill this in after implementing and testing both functions.*

**Test: what does the raw LLM response look like for one episode?**

```
Episode tested: Dr. Priya Nair on Adolescent Mental Health After the Pandemic
Raw response text:
LABEL: interview
REASONING: The host is speaking with a single guest (Dr. Priya Nair) in a structured question-and-response format to draw out her expertise and clinical practice findings.
```

**How did you parse the label out of the response?**

```
We split the raw response text by line, checked if a line starts with "LABEL:" (case-insensitively), extracted the trailing value, stripped surrounding whitespace, quotes, and punctuation (like dots), and then validated it against VALID_LABELS.
```

**Did any episodes return `"unknown"`? If so, why?**

```
None expected unless there is an API error or the LLM output fails to conform to the requested format.
```

**One thing about the output format that surprised you:**

```
The LLM occasionally includes a trailing period or quotes around the label name (e.g. `LABEL: "interview"` or `LABEL: interview.`), which makes strict regex matching fragile and highlights the need for stripping punctuation.
```
