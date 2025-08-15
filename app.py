import os
import json
import re
from typing import List, Dict, Any, Optional

import streamlit as st
from openai import OpenAI


# ---------------------------
# OpenAI Client Initialization
# ---------------------------
def get_openai_client() -> OpenAI:
    # OPENAI_API_KEY must be set in environment
    return OpenAI()


# ---------------------------
# Prompt Builders
# ---------------------------
def build_system_prompt() -> str:
    return (
        "You are an expert interviewer for top product companies like Flipkart, adept at generating MCQ-style quizzes "
        "for Data Structures and Algorithms with a focus on correctness and clarity.\n\n"
        "Output strictly valid JSON only (no code fences, no extra text, no markdown). Use this schema:\n"
        "{\n"
        '  "quiz_title": "string",\n'
        '  "topic_summary": "string",\n'
        '  "questions": [\n'
        "    {\n"
        '      "id": number,\n'
        '      "topic": "string",\n'
        '      "difficulty": "Easy" | "Medium" | "Hard",\n'
        '      "question": "string",\n'
        '      "options": ["string", "string", "string", "string"],\n'
        "      \"correct_index\": 0 | 1 | 2 | 3,\n"
        '      "explanation": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Produce exactly N questions as requested.\n"
        "- Each question must have exactly 4 distinct options (A/B/C/D order in array). Only one option is correct.\n"
        "- correct_index must be an integer 0..3.\n"
        "- Avoid ambiguity. No multiple-correct answers.\n"
        "- Focus on practical, Flipkart-style MCQs covering patterns, tradeoffs, and edge cases.\n"
        "- If code or outputs are referenced, keep them short and self-contained.\n"
        "- Keep explanations concise but sufficient to learn from.\n"
        "- No additional text beyond the JSON."
    )


def build_user_prompt(topics: List[str], difficulty: str, num_questions: int, include_graph_focus: bool) -> str:
    focus = (
        "Emphasize Graphs topics (BFS/DFS, topological sorting, shortest paths, MST, Union-Find, strongly connected components, DAG properties, graph traversal complexity, adjacency representations), "
        if include_graph_focus else ""
    )

    topic_line = (
        "All Data Structures (Arrays, Linked Lists, Stacks, Queues, Trees, Heaps, Hashing, Graphs, Tries, Dynamic Programming)"
        if not topics
        else ", ".join(topics)
    )

    return (
        f"Create a Flipkart-style MCQ quiz.\n"
        f"- Topics: {topic_line}\n"
        f"- Difficulty: {difficulty}\n"
        f"- Number of questions (N): {num_questions}\n"
        f"- {focus}"
        "Also cover: time and space complexity, edge cases, invariants, and when to choose one approach over another. "
        "Questions should be diverse (theory, output prediction on small inputs, data structure properties, algorithmic tradeoffs). "
        "Ensure JSON strictly follows the schema."
    )


# ---------------------------
# Utilities
# ---------------------------
def extract_json(text: str) -> Optional[dict]:
    """
    Try to robustly parse JSON from the model response.
    """
    text = text.strip()

    # Remove markdown-style code fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)

    # Quick direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback: attempt to find the largest JSON object
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            return json.loads(candidate)
    except Exception:
        return None

    return None


def validate_questions(data: dict, expected_count: int) -> List[Dict[str, Any]]:
    """
    Validate and normalize questions according to our UI expectations.
    """
    questions = data.get("questions", [])
    valid = []

    for q in questions:
        if not isinstance(q, dict):
            continue

        options = q.get("options", [])
        if (
            not isinstance(options, list)
            or len(options) != 4
            or not all(isinstance(opt, str) and opt.strip() for opt in options)
        ):
            continue

        correct_index = q.get("correct_index", None)
        if not isinstance(correct_index, int) or correct_index < 0 or correct_index > 3:
            continue

        entry = {
            "id": q.get("id", len(valid) + 1),
            "topic": q.get("topic", "DSA"),
            "difficulty": q.get("difficulty", "Medium"),
            "question": q.get("question", "").strip(),
            "options": options,
            "correct_index": correct_index,
            "explanation": q.get("explanation", "").strip(),
        }

        if entry["question"] and len(set(entry["options"])) == 4:
            valid.append(entry)

    # Trim or pad to expected_count (do not fabricate, just trim)
    return valid[:expected_count]


# ---------------------------
# OpenAI Interaction
# ---------------------------
def generate_mcqs(
    client: OpenAI,
    topics: List[str],
    difficulty: str,
    num_questions: int,
    temperature: float,
    model_name: str,
    include_graph_focus: bool,
) -> Dict[str, Any]:
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(topics, difficulty, num_questions, include_graph_focus)

    response = client.chat.completions.create(
        model=model_name,  # "gpt-4" or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    content = response.choices[0].message.content
    data = extract_json(content)

    if not data:
        raise ValueError("Failed to parse model output as JSON.")

    questions = validate_questions(data, num_questions)
    if not questions:
        raise ValueError("Received no valid questions from the model.")

    title = data.get("quiz_title", "Flipkart DSA MCQs")
    summary = data.get("topic_summary", "")
    return {"title": title, "summary": summary, "questions": questions}


# ---------------------------
# Session State Management
# ---------------------------
def init_state():
    if "quiz" not in st.session_state:
        st.session_state.quiz = None  # dict with title, summary, questions
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "answers" not in st.session_state:
        st.session_state.answers = {}  # index -> selected option idx
    if "revealed" not in st.session_state:
        st.session_state.revealed = {}  # index -> bool
    if "score" not in st.session_state:
        st.session_state.score = 0


def reset_quiz_state():
    st.session_state.current_index = 0
    st.session_state.answers = {}
    st.session_state.revealed = {}
    st.session_state.score = 0


# ---------------------------
# UI Components
# ---------------------------
def sidebar_controls() -> Dict[str, Any]:
    st.sidebar.header("Quiz Settings")

    available_topics = [
        "Graphs",
        "Arrays",
        "Linked Lists",
        "Stacks",
        "Queues",
        "Trees",
        "Heaps",
        "Hashing",
        "Tries",
        "Dynamic Programming",
    ]

    selected_topics = st.sidebar.multiselect(
        "Select Topics (leave empty for all)",
        options=available_topics,
        default=["Graphs"],
    )

    difficulty = st.sidebar.selectbox("Difficulty", options=["Easy", "Medium", "Hard"], index=1)
    num_questions = st.sidebar.slider("Number of Questions", min_value=3, max_value=20, value=8, step=1)

    model_name = st.sidebar.selectbox("Model", options=["gpt-4", "gpt-3.5-turbo"], index=0)
    temperature = st.sidebar.slider("Creativity (temperature)", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

    include_graph_focus = st.sidebar.checkbox("Emphasize Graphs", value=True)

    st.sidebar.markdown("---")
    st.sidebar.caption("Provide your OPENAI_API_KEY as an environment variable to run.")

    return {
        "topics": selected_topics,
        "difficulty": difficulty,
        "num_questions": num_questions,
        "model_name": model_name,
        "temperature": temperature,
        "include_graph_focus": include_graph_focus,
    }


def render_quiz_header(quiz_meta: Dict[str, Any]):
    st.subheader(quiz_meta.get("title", "Flipkart DSA MCQs"))
    if quiz_meta.get("summary"):
        st.caption(quiz_meta["summary"])


def render_question(
    q: Dict[str, Any],
    idx: int,
    show_explanation_when_revealed: bool = True,
):
    st.write(f"Q{idx + 1}. {q['question']}")
    selected = st.session_state.answers.get(idx, None)

    choice = st.radio(
        "Choose an option:",
        options=list(enumerate(q["options"])),
        index=selected if isinstance(selected, int) else 0,
        format_func=lambda t: f"{chr(65 + t[0])}. {t[1]}",
        key=f"radio_{idx}",
    )

    # Persist selection
    st.session_state.answers[idx] = choice[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Reveal Answer", key=f"reveal_{idx}"):
            st.session_state.revealed[idx] = True
            # Update score only on first reveal if correct
            if st.session_state.answers.get(idx) == q["correct_index"]:
                # Prevent double-counting
                already_counted_key = f"counted_{idx}"
                if not st.session_state.get(already_counted_key, False):
                    st.session_state.score += 1
                    st.session_state[already_counted_key] = True

    with col2:
        if st.button("Clear Choice", key=f"clear_{idx}"):
            st.session_state.answers[idx] = 0
            st.session_state.revealed[idx] = False
            st.session_state[f"counted_{idx}"] = False

    with col3:
        st.write("")

    if st.session_state.revealed.get(idx, False):
        correct_idx = q["correct_index"]
        if st.session_state.answers.get(idx) == correct_idx:
            st.success(f"Correct! Answer: {chr(65 + correct_idx)}. {q['options'][correct_idx]}")
        else:
            st.error(f"Incorrect. Correct answer: {chr(65 + correct_idx)}. {q['options'][correct_idx]}")

        if show_explanation_when_revealed and q.get("explanation"):
            st.info(f"Why: {q['explanation']}")


def render_navigation(total: int):
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Previous") and st.session_state.current_index > 0:
            st.session_state.current_index -= 1
    with col2:
        st.write(f"Question {st.session_state.current_index + 1} of {total}")
    with col3:
        if st.button("Next") and st.session_state.current_index < total - 1:
            st.session_state.current_index += 1


def render_scoreboard(total: int):
    answered_correct = 0
    for i in range(total):
        q = st.session_state.quiz["questions"][i]
        if st.session_state.answers.get(i) == q["correct_index"] and st.session_state.revealed.get(i, False):
            answered_correct += 1
    st.metric(label="Score (revealed & correct)", value=f"{answered_correct} / {total}")


# ---------------------------
# Main App
# ---------------------------
def main():
    st.set_page_config(page_title="Flipkart-style DSA MCQ Agent", page_icon="🧠", layout="centered")
    st.title("Flipkart-style DSA MCQ Agent")
    st.caption("Generate and practice MCQs on Graphs and Data Structures with explanations.")

    init_state()
    controls = sidebar_controls()

    # Action buttons
    col_gen, col_reset = st.columns(2)
    with col_gen:
        generate_clicked = st.button("Generate Quiz")
    with col_reset:
        reset_clicked = st.button("Reset Quiz")

    if reset_clicked:
        st.session_state.quiz = None
        reset_quiz_state()

    # Generate quiz on click
    if generate_clicked:
        try:
            client = get_openai_client()
            quiz = generate_mcqs(
                client=client,
                topics=controls["topics"],
                difficulty=controls["difficulty"],
                num_questions=controls["num_questions"],
                temperature=controls["temperature"],
                model_name=controls["model_name"],
                include_graph_focus=controls["include_graph_focus"],
            )
            st.session_state.quiz = quiz
            reset_quiz_state()
        except Exception as e:
            st.error(f"Failed to generate quiz: {e}")

    # Display quiz
    if st.session_state.quiz:
        render_quiz_header(st.session_state.quiz)
        total_q = len(st.session_state.quiz["questions"])
        if total_q == 0:
            st.warning("No questions available. Try generating again.")
            return

        # Scoreboard
        render_scoreboard(total_q)
        st.markdown("---")

        # Current question
        idx = st.session_state.current_index
        q = st.session_state.quiz["questions"][idx]
        render_question(q, idx, show_explanation_when_revealed=True)

        st.markdown("---")
        render_navigation(total_q)

        # Download buttons
        st.subheader("Export")
        json_data = json.dumps(st.session_state.quiz, indent=2)
        st.download_button(
            label="Download Quiz JSON",
            data=json_data,
            file_name="flipkart_dsa_mcqs.json",
            mime="application/json",
        )
    else:
        st.info("Click 'Generate Quiz' to create a new set of MCQs.")


if __name__ == "__main__":
    main()