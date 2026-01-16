# ai-in-market-trend-analysis 


## Early Panic Detection in Gold Markets Using News Sentiment and Price Volatility
Done for the attainment of the AI Minor degree from IIT Ropar. Performed under the guidance of Dr.Niranjan Deshpande


### Project Overview

This project implements a **hybrid AI framework** to detect **panic-driven volatility** in the gold market by combining:

1. **Time-Series Signal Analysis**  
   - Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN)  
   - Hilbert–Huang Transform (HHT) for instantaneous volatility extraction  

2. **Natural Language Processing (NLP)**  
   - Bi-directional LSTM (Bi-LSTM) network for financial news sentiment classification  
   - Classes: Negative, Neutral, Positive  

By **fusing market volatility and negative news sentiment**, the system generates **panic alerts** to identify potential extreme market movements.  

### Getting Started

#### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/ai-in-market-trend-analysis.git
cd gold-panic-detection
```
#### 2. Install Dependecies 
Using pip:
```bash
pip install -r requirements.txt
```
Or Conda:
```bash
conda env create -f environment.yml
conda activate gold-market-volatility-env
```
Key Libraries are:
1. numpy==1.26.x
2. pandas
3. yfinance
4. PyEMD
5. tensorflow
6. scikit-learn
7. matplotlib
8. seaborn

#### 3. Data Preparation
1. Place raw datasets in data/raw/
2. Run the file to process and save the data in data/processed

#### 4. Execution Instructions
1. Open the .ipynb file in Jupyter or Google Colab
2. Run all cells top-to-bottom for reproducibility
3. Ensure the data/ folder contains the required datasets (gold price and news CSV)

#### 5. Results Summary

1. Sentiment Classification Accuracy: 88%
   Negative: 92% precision, 90% recall
   Neutral: 81% precision, 76% recall
   Positive: 89% precision, 94% recall
3. Panic Alert Detection: Correctly flagged historical high-volatility periods (e.g., 2008–2009)
4. Visualizations:
   Gold price vs. HHT volatility
   Panic alert markers
   Confusion matrix for sentiment classification
#### 6. Notes and Recommendations 
Designed for educational and research purposes, not for live trading
CEEMDAN may require NumPy < 2.0 for compatibility
To retrain the Bi-LSTM model, adjust hyperparameters directly in the notebook

#### 7. References and Data Sources

1. Yahoo Finance (Gold Prices): GC=FGold News Dataset: gold_news.csv

Citations:

1. Sinha, Ankur, and Tanmay Khandait. "Impact of News on the Commodity Market: Dataset and Results." *Future of Information and Communication Conference*, pp. 589–601. Springer, Cham, 2021. [DOI/Publisher Link](https://www.sciencedirect.com/science/article/abs/pii/S1062940821000553)  
   - The dataset and methodology inspired the news sentiment analysis in this project.

2. Sinha, Ankur, and Tanmay Khandait. "Impact of News on the Commodity Market: Dataset and Results." *arXiv preprint arXiv:2009.04202*, 2020. [arXiv Link](https://arxiv.org/abs/2009.04202)  
   - Used as a reference for constructing the gold news dataset and labeling strategy.

3. Ozupek, O.; Yilmaz, R.; Ghasemkhani, B.; Birant, D.; Kut, R.A. "A Novel Hybrid Model (EMD-TI-LSTM) for Enhanced Financial Forecasting with Machine Learning." *Mathematics*, 2024, 12, 2794. [DOI Link](https://doi.org/10.3390/math12172794)  
   - Inspired the hybrid CEEMDAN + Bi-LSTM approach for time series and sentiment fusion.

4.  Wu, Z., & Huang, N.E. "Ensemble Empirical Mode Decomposition: A Noise-Assisted Data Analysis Method." *Mechanical Systems and Signal Processing*, 2009, 24(4), 661–680. [DOI Link](https://doi.org/10.1016/j.ymssp.2009.04.006)  
   - Provides the theoretical foundation for CEEMDAN used in signal decomposition of gold prices.


Libraries: PyEMD, TensorFlow, scikit-learn

#### License & AI Usage

License: MIT

AI Tools Used: Python libraries; no generative AI was used in dataset creation or analysis
   




