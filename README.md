📊 Bangla Social Media Sentiment & Emotion Dynamics (July 2024 Protest)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NLP](https://img.shields.io/badge/NLP-Transformers_%7C_BanglaBERT-blue?style=for-the-badge)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-TCN_%7C_LSTM-orange?style=for-the-badge)
![Stats](https://img.shields.io/badge/Statistical_Analysis-Survival_Analysis_%7C_Markov-brightgreen?style=for-the-badge)

📌 Project Overview (Bachelor Thesis)
This repository encompasses the technical methodology and implementation for my Bachelor’s thesis, titled "Time-Series Forecasting and Emotion Transition Analysis of Bangla Social Media Sentiment." 

The project establishes an end-to-end computational pipeline designed to analyze collective emotional dynamics in Bangladesh during the July 2024 protests. By processing a corpus of over 16,000 raw Bangla Facebook comments, the research utilizes a fine-tuned, multi-task `Bangla-BERT` model for sentiment, emotion, and topic classification. Furthermore, it benchmarks advanced deep sequence architectures—specifically Temporal Convolutional Networks (TCN) and Long Short-Term Memory (LSTM) networks—against traditional statistical models, such as SARIMA and Prophet, for time-series anomaly detection.

🏆 Key Results & Contributions

* **Low-Resource NLP:** Fine-tuned `Bangla-BERT` for multi-task classification, achieving 78.2% sentiment and 74.7% topic accuracy on noisy, code-mixed text.
* **Deep Sequence Forecasting:** TCN architectures outperformed LSTM, Prophet, and SARIMA in non-linear sentiment forecasting (TCN MAE = 0.70).
* **Anomaly Detection:** Residual thresholding on TCN forecasts successfully flagged ground-truth protest milestones with up to 50% precision.
* **Behavioral Statistical Modeling:** 
    * Proven strong topic-emotion correlation ($\chi^2 \approx 7486, p < 0.001$).
    * Kaplan-Meier analysis confirmed "Anger" had the longest median survival time (~5 hours) in continuous discourse.

🗂️ Repository Structure & Notebooks
The project workflow is divided into modular Jupyter Notebooks:
`NLP_Preprocessing.ipynb`: Data engineering pipeline. Handles URL/digit removal, non-Bangla character filtering, diacritic normalization, and stop-word filtering using `bnltk`.
`labeling.ipynb`: Contains the multi-task `Bangla-BERT` architecture. Includes the custom training loop (AdamW optimizer, linear warmup/decay) used to auto-label $\approx 11,000$ comments based on a manually annotated seed set.
`Sentiment_Intensity_Shift.ipynb` & `sentiment_shift_OverTime.ipynb`: Time-series generation and forecasting. Benchmarks SARIMA, Prophet, LSTM (7-day look-back), and TCN (dilated causal convolutions).
`EmotionDynamics.ipynb`: Advanced statistical analysis. Includes K-means clustering of hourly sentiment, $6 \times 6$ Emotion Transition Probability Matrices (Markov chains), Chi-square testing, and Kaplan-Meier survival curves.

🛠️ Tech Stack & Methodology
Language: Python
NLP & Deep Learning: HuggingFace Transformers (`sagorsarker/bangla-bert-base`), TensorFlow/Keras (TCN, LSTM).
Time-Series & Stats: `statsmodels` (SARIMA), `prophet`, `scikit-learn` (K-Means, metrics), `lifelines` (Kaplan-Meier survival analysis).
Data Processing: Pandas, NumPy, `bnltk`.

🚀 Installation & Usage
To replicate this environment and run the notebooks:
```bash
# Clone the repository
git clone https://github.com/rahuldgofficial/Thesis.git
cd Thesis

# Create a virtual environment and install dependencies
python -m venv thesis_env
source thesis_env/bin/activate  # On Windows use: thesis_env\Scripts\activate
pip install -r requirements.txt # Ensure transformers, tensorflow, and stats libraries are installed
```

📊 Visual Highlights

1. TCN Sentiment Forecasting vs Actual Baseline
   <img width="400" height="250" alt="daily sentiment intensity_ forecast with 95% CIs" src="https://github.com/user-attachments/assets/9b542b1c-e08a-42b7-9537-dc517031edd0" />
   
2. Emotion Transition Probability Matrix
   <img width="400" height="250" alt="emotion transition probabilities" src="https://github.com/user-attachments/assets/87cd21e1-105b-4cb7-8398-3b4714cd8e7a" />
   
3. Emotion State Survival Analysis (Kaplan-Meier)
  <img width="400" height="250" alt="Emotion state survival analysis" src="https://github.com/user-attachments/assets/697d0264-6093-4855-8c6e-5fe56edc442e" />
  
4. TCN based threshold sensitivity
  <img width="400" height="250" alt="TCN based threshold sensitivity" src="https://github.com/user-attachments/assets/93be8be3-5197-44cb-8dd7-039f2608629b" />
  
  
5. TCN Forecast & Anomaly Detection
  <img width="400" height="250" alt="TCN forecast   anomaly detection" src="https://github.com/user-attachments/assets/b091d4b6-d87c-4a48-a857-057fafedd4dc" />
  
6. Anomaly Detection via LSTM Residuals
  <img width="400" height="250" alt="anomaly detection via LSTM Residuals" src="https://github.com/user-attachments/assets/e10bc115-6d9b-40b1-b457-12ed7a81dc28" />
  
7. Change Point Detection on Sentiment Intensity
  <img width="400" height="250" alt="change point detection on sentiment intensity" src="https://github.com/user-attachments/assets/3a8ff35e-9d4a-414c-a298-11ef69b12e58" />
  
8. Topic Emotion Associations
  <img width="400" height="250" alt="topic emotion associations" src="https://github.com/user-attachments/assets/b13375a5-5a95-4e15-8327-f9bc897a7148" />
  
9. Daily Sentiment Clusters
  <img width="400" height="250" alt="daily sentiment clusters" src="https://github.com/user-attachments/assets/ab6e8e35-da4f-489a-a590-4f85844b9870" />
  
10. Forecast MAE
  <img width="400" height="250" alt="forecast MAE 0 890" src="https://github.com/user-attachments/assets/d26956d0-2ad7-47f9-958a-3fb3a07e6532" />
  
11. Emotion Dynamics Timeline
  <img width="400" height="250" alt="emotion dynamics timeline" src="https://github.com/user-attachments/assets/603016bb-dd53-445e-8114-c594b85f3b4a" />
  
12. Hourley Sentiment Heat map
  <img width="400" height="250" alt="hourley sentiment heat map" src="https://github.com/user-attachments/assets/fc85638a-c0ac-4dd9-a394-b2e7bf70c9dc" />
  
13. Sentiment Intensity Actual vs Forecast
  <img width="400" height="250" alt="sentiment_intensity Actual vs Forecast" src="https://github.com/user-attachments/assets/e6f91fa5-47b8-4d83-ba3a-fe3c0109b9df" />
    
📄 Read Full Manuscript Here
https://doi.org/10.31235/osf.io/qsvrm_v1
   
