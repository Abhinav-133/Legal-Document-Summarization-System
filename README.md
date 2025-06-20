# 🧾 Legal Document Summarization System

This project focuses on generating high-quality summaries of legal case documents using both **extractive** and **abstractive** summarization techniques. It assists legal professionals, researchers, and students in quickly understanding lengthy legal texts by providing concise and coherent summaries while preserving essential legal information.

---

## 🚀 Features

- 📚 Supports both **abstractive** and **extractive** summarization methods.
- 🧠 Uses state-of-the-art transformer models like **Pegasus** and **LED**.
- 📝 Custom extractive models including **MMR** and **CasseSummarizer**.
- ⚖️ Integrates **DelSumm**, a hybrid ensemble optimized for legal summaries.
- 📊 Outputs are evaluated for coherence, legal fidelity, and readability.

---

## 🧠 Algorithms Used

| Type          | Method              | Description |
|---------------|---------------------|-------------|
| **Abstractive** | **Pegasus**          | Transformer-based model pre-trained for summarization tasks, fine-tuned on legal documents. |
|                | **LED (Longformer Encoder Decoder)** | Efficiently handles long inputs using sparse attention mechanisms. |
| **Extractive**  | **MMR (Maximal Marginal Relevance)** | Selects diverse and relevant sentences to reduce redundancy. |
|                | **CasseSummarizer**   | Extracts based on rhetorical roles and custom legal heuristics. |
| **Hybrid**      | **DelSumm**           | Combines deep models and legal rules for domain-specific summarization. |

---

## 📂 Project Structure

```
legal-summarization/
├── data/
│   └── sample_cases/
├── models/
│   ├── pegasus.py
│   ├── led.py
│   ├── mmr.py
│   ├── casse.py
│   └── delsumm.py
├── utils/
│   └── preprocess.py
├── evaluation/
│   └── metrics.py
├── main.py
└── README.md
```

---

## 🔧 Installation

```bash
git clone https://github.com/yourusername/legal-summarizer.git
cd legal-summarizer

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Usage Example

```python
# Example: Abstractive Summarization using Pegasus
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large")

input_text = "Your legal document text goes here..."

inputs = tokenizer(input_text, truncation=True, padding="longest", return_tensors="pt")
summary_ids = model.generate(**inputs, max_length=256, num_beams=5, early_stopping=True)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Summary:", summary)
```

You can similarly run **LED**, **MMR**, **CasseSummarizer**, or **DelSumm** by executing their respective scripts inside the `/models` directory.

---

## 📊 Evaluation Metrics

- **ROUGE-1 / ROUGE-2 / ROUGE-L**
- **BERTScore**
- **Coverage and Redundancy**
- **Domain-specific Legal Fidelity Score** *(custom metric)*

---

## 🧠 Future Improvements

- Add rhetorical role-aware summarization enhancement
- Benchmark on open legal datasets like **CUAD**, **ECtHR**, and **INDIANJ**
- Integrate summarization into a legal search pipeline
