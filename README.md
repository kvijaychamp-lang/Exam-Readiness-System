# AI Exam Readiness Analyzer

> A professional-grade, AI-powered academic evaluation tool built with Python and Streamlit.

---

## Overview

The **AI Exam Readiness Analyzer** is a semantic evaluation platform designed to help students objectively measure how well their answers align with reference material. It goes beyond simple keyword matching by using state-of-the-art Sentence Transformer models to compute deep semantic similarity, giving students actionable, data-driven feedback before their exams.

---

## Key Features

| Feature | Description |
|---|---|
| **Semantic Similarity** | Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to compute deep contextual similarity between answers |
| **PDF Reference Support** | Upload multi-page PDF files as the reference answer with a real-time page-by-page progress bar |
| **Subject-Aware Evaluation** | Adjustable strictness thresholds for Technical/Science, Theory/Arts, and Coding subjects |
| **Radar / Spider Chart** | Plotly radar chart visualising four metrics: Semantic Accuracy, Keyword Coverage, Sentence Clarity, and Concept Depth |
| **Readiness Gauge** | Speedometer-style gauge chart showing the overall readiness score with colour-coded zones |
| **Answer Highlighter** | User's response rendered with green-highlighted matched keywords, dimmed filler words, and normal content words |
| **Viva Mode** | Speech-to-text input via microphone using `streamlit-mic-recorder` for spoken answer practice |
| **Progress Trend** | Session-state line chart tracking improvement across the last 3 evaluation attempts |
| **Missing Keywords** | Monospace badge display of technical terms present in the reference but absent from the user's answer |
| **CSV Export** | Download a structured analysis report containing score, metrics, missing keywords, and recommendations |
| **Text Export** | Download a plain-text summary report of the full evaluation |
| **Professional UI** | Dark-themed, GitHub-style interface with no distracting animations |

---

## Project Structure

```
ai-exam-readiness-analyzer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # Project documentation
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-exam-readiness-analyzer.git
cd ai-exam-readiness-analyzer
```

### 2. Create and activate a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## How to Use

1. **Select a Subject Category** from the sidebar (Technical/Science, Theory/Arts, or Coding).
2. **Provide the Reference Answer** — paste text directly or upload a PDF.
3. **Enter Your Answer** — type it in the text area or use Viva Mode to speak it.
4. Click **Run Evaluation** to generate the full analysis.
5. Review your **Readiness Score**, **Radar Chart**, **Missing Keywords**, and **Recommendations**.
6. **Export** your results as a `.csv` or `.txt` report.

---

## Tech Stack

- **[Streamlit](https://streamlit.io/)** — Web application framework
- **[Sentence Transformers](https://www.sbert.net/)** — Semantic similarity model
- **[Plotly](https://plotly.com/python/)** — Interactive charts (Gauge, Radar, Bar, Line)
- **[PyPDF2](https://pypdf2.readthedocs.io/)** — PDF text extraction
- **[streamlit-mic-recorder](https://github.com/B4PT0R/streamlit-mic-recorder)** — Browser-based speech-to-text

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Author

Built as a student project to demonstrate applied NLP, data visualisation, and full-stack Python development.
