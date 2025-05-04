<h1 align="center" id="title">Ai_Huma_Text_Classification</h1>

<p id="description">Using the Hugging Face Transformers library a RoBERTa-based model is used to distinguish between human-written and AI-generated text. The app provides a user-friendly interface for real-time predictions. The model is loaded either from a local directory or directly from the Hugging Face Model Hub.</p>

<h2>🛠️ Installation Steps:</h2>

<p>1. 1. Clone the repository</p>

```
git clone https://github.com/Abuzaid-01/ai_human_text_detector.git cd ai_human_text_detector
```

<p>2. 2. Create and activate a virtual environment</p>

```
python -m venv venv source venv/bin/activate        # On Windows: venv\Scripts\activate
```

<p>3. 3. Install dependencies</p>

```
pip install streamlit torch transformers
```

<p>4. 4. Run the Streamlit app bash Copy Edit</p>

```
streamlit run app.py
```

<p>5. Hugging Face Model</p>

```
 https://huggingface.co/Abuzaid01/Ai_Human_text_detect
```

<p>6. You can load it in Python using</p>

```
from transformers import RobertaForSequenceClassification RobertaTokenizerFast  model = RobertaForSequenceClassification.from_pretrained("Abuzaid-01/ai-vs-human-text-classifier") tokenizer = RobertaTokenizerFast.from_pretrained("Abuzaid-01/ai-vs-human-text-classifier")
```

