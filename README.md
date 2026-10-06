Here is a comprehensive, production-grade **`README.md`** tailored for your project repository. It includes all architecture details, dataset specs, training logs across epochs, API/UI setup instructions, and project structure.

---

# 🔤 Seq2Seq Transformer English-to-Italian NMT

An end-to-end Neural Machine Translation (NMT) pipeline implementing a custom **Sequence-to-Sequence Transformer** architecture built from scratch in **PyTorch** (based on *Attention Is All You Need*). The project is fully packaged and served via a **FastAPI** backend and an interactive **Streamlit** dashboard.

---

## 📌 Project Overview

This project implements a complete translation system that converts **English** source sentences into **Italian** target sentences.

* **Model:** Built from scratch using PyTorch (Multi-Head Self-Attention, Masked Attention, Positional Encoding, and Feed-Forward Networks).
* **Dataset:** Trained on the `opus_books` parallel English-Italian corpus.
* **Deployment:** Integrated REST API via **FastAPI** with custom auto-regressive greedy decoding and a clean **Streamlit** user interface.

---

## 🏗️ Architecture Specifications

```text
Input Tokens -> Input Embeddings -> Positional Encoding 
                 |
                 v
        +------------------+
        | Encoder (x6)     | -> Multi-Head Self Attention + FFN + Residual/LayerNorm
        +------------------+
                 |
                 v
        +------------------+
        | Decoder (x6)     | -> Masked Self Attention + Cross Attention + FFN
        +------------------+
                 |
                 v
        Projection Layer -> Vocabulary Logits -> Greedy Decoding

```

| Hyperparameter / Feature | Specification |
| --- | --- |
| **Model Dimensions ($d_{model}$)** | `512` |
| **Attention Heads ($h$)** | `8` |
| **Encoder / Decoder Layers** | `6` / `6` |
| **Feed-Forward Inner Dim ($d_{ff}$)** | `2048` |
| **Sequence Length (`seq_len`)** | `350` |
| **Dropout** | `0.1` |
| **Source Vocabulary Size** | `15,698` (English) |
| **Target Vocabulary Size** | `22,463` (Italian) |

---

## 📊 Training Details & Performance Across Epochs

The model was trained on an **NVIDIA Tesla T4 GPU** (16GB VRAM) for **5 Epochs** before inference evaluation.

### Epoch Summary

* **Epoch 00:** `Loss: 5.747` — *Initial learning phase; model output shows severe token repetition (`Non mi a , e mi...`).*
* **Epoch 01:** `Loss: 4.247` — *Model starts learning phrase structure (`a bordo di...`).*
* **Epoch 02:** `Loss: 5.563` — *Captures room and subject context, but long-range dependencies produce repetition loops.*
* **Epoch 03:** `Loss: 4.663` — *Outputs short dialogue and conversational phrases accurately.*
* **Epoch 04:** `Loss: 4.785` — *Final saved checkpoint (`tmodel_04.pt`); produces clean, coherent translations for simple/medium sequence lengths.*

### Sample Validation Outputs (Epoch 04 Checkpoint)

| Source (English) | Target (Italian) | Model Prediction |
| --- | --- | --- |
| *Dolly did not reply.* | *Dar’ja Aleksandrovna non obiettava nulla.* | **Dolly non rispose .** |
| *'Well, where are they?'* | *— Su, dove sono?* | **— E allora , dove sono ?** |
| *'You are old, Father William,' the young man said,* | *“Sei vecchio, caro babbo” — gli disse il ragazzino —* | **— Voi siete un vecchio , — disse il vecchio .** |

---

## 📂 Project Directory Structure

```text
TransformerArch/
│
├── weights/
│   └── tmodel_04.pt            # Downloaded PyTorch model checkpoint
│
├── tokenizer_en.json           # English WordPiece/BPE tokenizer
├── tokenizer_it.json           # Italian WordPiece/BPE tokenizer
│
├── config.py                   # Hyperparameters and path configurations
├── model.py                    # PyTorch Transformer architecture definition
├── dataset.py                  # Dataset loader & tokenization logic
│
├── api.py                      # FastAPI REST server with greedy decoding endpoint
├── app.py                      # Streamlit frontend user interface
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

```

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites & Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Xaayu/TransformerArch.git
cd TransformerArch

pip install -r requirements.txt

```

### 2. Required Dependencies (`requirements.txt`)

```text
torch
tokenizers
torchmetrics
fastapi
uvicorn
streamlit
pydantic
requests

```

---

## 💻 Running the Application

### Step 1: Start the FastAPI Backend

Launch the API server using Uvicorn:

```bash
uvicorn api:app --reload --port 8000

```

* Interactive API Documentation (Swagger UI) will be available at: `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)`

### Step 2: Start the Streamlit Frontend UI

In a separate terminal window, launch the web dashboard:

```bash
streamlit run app.py

```

* Access the translation UI in your browser at: `http://localhost:8501`

---

## 📡 API Reference

### `POST /translate`

Translates an English sentence into Italian.

* **Request Body:**
```json
{
  "text": "Hello, how are you today?"
}

```


* **Response:**
```json
{
  "source_text": "Hello, how are you today?",
  "translated_text": "Ciao , come stai oggi ?"
}

```



---

## 🛠️ Future Improvements

* **Inference Strategy:** Implement **Beam Search Decoding** with a length penalty to eliminate repetition loops on long-sequence generation.
* **Dataset Scaling:** Train on larger datasets (e.g., Europarl or OPUS-100) for increased vocabulary coverage.
* **Learning Rate Scheduling:** Apply warm-up learning rate decay to improve convergence stability.
