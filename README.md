### README.md 

```markdown
#  Employee Sentiment Analysis & Forecasting

This project analyzes internal employee messages to understand sentiment trends, identify key contributors, and forecast future sentiment using machine learning and time-series techniques.

---

##  Project Highlights

| Task | Summary |
|------|---------|
| **1. Sentiment Labeling** | Used `cardiffnlp/twitter-roberta-base-sentiment` to classify messages as Positive, Neutral, or Negative. Added model confidence scores for scoring. |
| **2. Exploratory Data Analysis** | Identified sentiment trends over time and across employees. Found recurring cycles of negative sentiment. |
| **3. Employee Scoring** | Mapped sentiments to numeric scores and aggregated by employee/month. |
| **4. Employee Ranking** | Top 3 and bottom 3 employees listed monthly based on sentiment scores. |
| **5. Flight Risk Detection** | Used rolling windows to flag employees with ≥4 negative messages in 30 days. |
| **6. Forecasting** | Compared **Linear Regression + Lasso** with **Prophet + Lasso regressors** to predict future sentiment scores. |

---

##  Sample Output

**Top 3 Positive Employees**
```

month      from                         score
2010-03    [sally.beck@enron.com](mailto:sally.beck@enron.com)         6
2010-04    [kayne.coulter@enron.com](mailto:kayne.coulter@enron.com)      5
2010-02    [bobette.riner@ipgdirect.com](mailto:bobette.riner@ipgdirect.com)  4

```

**Top 3 Negative Employees**
```

month      from                         score
2010-01    [sally.beck@enron.com](mailto:sally.beck@enron.com)        -1
2010-01    [bobette.riner@ipgdirect.com](mailto:bobette.riner@ipgdirect.com)  0
2010-01    [john.arnold@enron.com](mailto:john.arnold@enron.com)        0

```

**Flight Risk Employees**
```

[bobette.riner@ipgdirect.com](mailto:bobette.riner@ipgdirect.com)
[john.arnold@enron.com](mailto:john.arnold@enron.com)
[don.baughman@enron.com](mailto:don.baughman@enron.com)
[sally.beck@enron.com](mailto:sally.beck@enron.com)
[rhonda.denton@enron.com](mailto:rhonda.denton@enron.com)

```

---

## Forecasting Comparison

| Model                      | MSE     | MAE     | Interpretability         |
|---------------------------|---------|---------|--------------------------|
| Linear Regression + Lasso | 0.1139  | -       | Simple, fast, explainable |
| Prophet + Lasso           | 0.0362  | 0.1561  | Trend + seasonality aware |

---

## Tech Stack

- Python
- RoBERTa (HuggingFace Transformers)
- Scikit-learn (Lasso, Linear Regression)
- Facebook Prophet
- Pandas, NumPy, Matplotlib, Seaborn

---

## Key Skills Demonstrated

- Sentiment Analysis using Transformers
- Feature Engineering (rolling stats, Lasso selection)
- Time-series Forecasting (Regression & Prophet)
- Risk Flagging using temporal logic
- Data Visualization & EDA

---

## Summary 

**Sentiment Forecasting Project**  
Built an interpretable system for analyzing internal messages using RoBERTa sentiment classification, feature-selected forecasting models (Lasso + Regression/Prophet), and risk identification. Achieved high forecast accuracy (MSE = 0.0362) using Prophet with rolling statistical regressors.
```

---

 `requirements.txt` file content:

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
prophet
transformers
torch

