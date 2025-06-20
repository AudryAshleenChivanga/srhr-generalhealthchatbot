#  Project Overview: SRHR & General Health Chatbot using DistilGPT-2 (Ashlebot)

In this project, I developed a domain-specific chatbot focused on **Sexual and Reproductive Health and Rights (SRHR)** and **general medical topics** using a fine-tuned version of **DistilGPT-2**. The chatbot is designed to provide informative, contextually relevant, and user-friendly responses to health-related questions, addressing the knowledge gap often found in mainstream medical datasets—especially around SRHR topics.

---
### Click Image to view Chatbot's Full Interface

<p align="center">
  <a href="https://youtu.be/gGY2CUTfuis">
    <img src="./images/ashlebot.jpg" alt="AshleMedBot Demo" width="500">
  </a>
</p>

## Datasets Used

### 1. SRHR Custom Dataset
Upon exploring multiple public medical datasets, I identified a significant underrepresentation of SRHR content. To address this, I manually collected, cleaned, and structured a custom dataset containing over 300 QA pairs specific to SRHR topics, such as contraception, menstruation, reproductive rights, STIs, and gender-based health challenges. This dataset was tailored for relevance, language accessibility, and coverage of community-level SRHR concerns.

### 2. MedQuAD Dataset
- **Source:** [MedQuAD - Medical Question Answering Dataset](https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research)
- **File:** `medquad.csv`  
This dataset provides curated QA pairs from trusted medical sources like NIH and MedlinePlus, covering a wide range of general health conditions. It served to complement the SRHR dataset by grounding the chatbot in verified, evidence-based general medical knowledge.

---
###  Preprocessing:

- **Data Integration:**
  - Combined the custom SRHR dataset and the MedQuAD dataset into a unified DataFrame
- **Cleaning Steps:**
  - Removed unwanted characters, extra spaces, and HTML tags
  - Lowercased all text for consistency
  - Removed duplicated question-answer pairs
- **Formatting for Fine-Tuning:**
  - Each entry was structured as:  
    `question <|sep|> answer`
  - Tokenized using the tokenizer with:
    - `padding='max_length'`
    - `truncation=True`
    - `max_length` between 128–256 tokens
  - Encoded inputs as TensorFlow-friendly tensors
  - Split into training and validation sets

---
##  Model Development and Fine-Tuning

- **Base Model:** `distilgpt2` from Hugging Face Transformers (a smaller, faster GPT-2 variant)
- **Tokenizer:** AutoTokenizer with added `<|sep|>` separator to distinguish questions and answers
- **Preprocessing:**
  - Combined both datasets and cleaned text (punctuation, casing, duplicates)
  - Tokenized using `padding`, `truncation`, and max length control (128–256 tokens)
- **Training:**
  - Used TensorFlow backend (`TFAutoModelForCausalLM`)
  - Ran multiple experiments adjusting learning rate, batch size, sequence length, and epochs
  - Final model trained for 3 epochs on a batch size of 4 with a learning rate of `5e-5`
- **Evaluation:**
  - Assessed with BLEU score and perplexity
  - Observed a perplexity score of `3.21`, indicating solid fluency
  - BLEU scores were low due to lexical differences, though responses aligned contextually
- **Observations:**
  - SRHR questions triggered more relevant outputs due to the targeted dataset
  - Some responses showed repetition, suggesting further fine-tuning potential

---

##  Deployment Steps

1. **Model Saving:**
   - Saved the tokenizer and fine-tuned DistilGPT-2 model to the `distilgpt2_srhr_generalhealth_v1` directory
   - Compressed and exported the model for local deployment

2. **Deployment Interface:**
   - Built an interactive frontend using **Streamlit**, allowing real-time question-answer interaction
   - Hosted locally (and compatible with cloud platforms such as Hugging Face Spaces or Streamlit Sharing)



3. **Usage Instructions:**
   - Clone the repository
   - Load the fine-tuned model and tokenizer
   - Run the app via `streamlit run app.py` to start the chatbot interface

 
## Repository Structure

```
srhr-generalhealthchatbot/
│
├── app.py                  # Streamlit app source code
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── data/                   # Datasets and resources
├── notebook/               # Jupyter notebooks and experiments
└── gitignore               # Contains ignored variables and large files like my model 
```
---

## Impact

This project not only bridges a critical gap in SRHR data accessibility but also demonstrates how combining domain-specific and general health knowledge can lead to more inclusive AI solutions. It holds promise for community health education, chatbot-assisted clinics, and awareness campaigns focused on SRHR and beyond.
