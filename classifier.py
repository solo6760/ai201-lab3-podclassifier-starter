import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str, prompt_mode: str = "default", target_class: str = None) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.
    Supports experimental variations using prompt_mode and target_class.
    """
    # Clone and prepare examples based on prompt_mode
    examples = list(labeled_examples)
    
    if prompt_mode == "order_reverse":
        examples = examples[::-1]
    elif prompt_mode == "target_last" and target_class:
        # Move target class examples to the end of the few-shot list
        target_examples = [ex for ex in examples if ex.get("label") == target_class]
        other_examples = [ex for ex in examples if ex.get("label") != target_class]
        examples = other_examples + target_examples

    # Customize definitions if using detailed_instructions mode
    narrative_def = "narrative: A story is told over the course of the episode, usually with reporting, production, or multiple sources woven together. Structured as a story, not a conversation."
    if prompt_mode == "detailed_instructions" and target_class == "narrative":
        narrative_def = (
            "narrative: Reported or documentary storytelling. Follows a story, character, or event across time. "
            "Evidence is assembled from multiple sources. The episode has a story arc, not a conversation arc. "
            "Key signals include past tense, scenes, phrases like 'This episode follows...', 'Reported over X months...', "
            "or structural storytelling using external archives/interviews rather than a simple 1-on-1 interview conversation."
        )

    prompt_lines = [
        "You are an expert podcast episode classifier. Your task is to classify the format of a podcast episode based on its description.",
        "",
        "Use exactly one of the following four valid labels:",
        "- interview: A host speaks with one or more guests. Structured around questions and responses. The guest has expertise, experience, or a story that the host is drawing out.",
        "- solo: One host speaks alone, without guests. Could be a personal essay, opinion piece, tutorial, reflection, or walkthrough.",
        "- panel: Three or more speakers discuss a topic together without a clear host-guest dynamic. All participants contribute as rough equals.",
        f"- {narrative_def}",
        "",
        "Here are some labeled examples for context:",
        ""
    ]

    for example in examples:
        prompt_lines.extend([
            f"Title: {example.get('title', 'Unknown')}",
            f"Description: {example.get('description', '')}",
            f"LABEL: {example.get('label')}",
            "CONFIDENCE: 10",
            "---"
        ])

    prompt_lines.extend([
        "Now classify the following new episode description:",
        f"Description: {description}",
        "",
        "You must respond in exactly this format (do not include any conversational filler or other text):",
        "LABEL: <one of: interview, solo, panel, narrative>",
        "CONFIDENCE: <an integer from 1 to 10 reflecting your certainty in this classification, where 10 is absolute certainty and 1 is pure guess>",
        "REASONING: <a brief 1-2 sentence explanation of why this label applies, citing specific aspects of the description>"
    ])

    return "\n".join(prompt_lines)


def classify_episode(description: str, labeled_examples: list[dict], prompt_mode: str = "default", target_class: str = None) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.
    """
    try:
        # Step 1: Build the prompt
        prompt = build_few_shot_prompt(labeled_examples, description, prompt_mode=prompt_mode, target_class=target_class)

        # Step 2: Send it to the LLM
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0,
        )
        
        response_text = response.choices[0].message.content
        print(f"--- RAW LLM RESPONSE ---\n{response_text}\n-----------------------")

        # Step 3: Parse the response
        label = "unknown"
        confidence = 5
        reasoning = ""
        
        for line in response_text.splitlines():
            line_str = line.strip()
            if line_str.upper().startswith("LABEL:"):
                label_val = line_str[len("LABEL:"):].strip().lower()
                label_val = label_val.strip("\"'.,;` ")
                label = label_val
            elif line_str.upper().startswith("CONFIDENCE:"):
                conf_val = line_str[len("CONFIDENCE:"):].strip()
                conf_val = conf_val.strip("\"'.,;` ")
                try:
                    confidence = int(conf_val)
                except ValueError:
                    confidence = 5
            elif line_str.upper().startswith("REASONING:"):
                reasoning_val = line_str[len("REASONING:"):].strip()
                reasoning = reasoning_val

        if not reasoning:
            reasoning = response_text.strip()

        # Step 4: Validate the label
        if label not in VALID_LABELS:
            label = "unknown"

        return {
            "label": label,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    except Exception as e:
        # Step 5: Handle errors gracefully
        return {
            "label": "unknown",
            "confidence": 0,
            "reasoning": f"Error during classification: {str(e)}",
        }
