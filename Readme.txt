===========================
Task 1: Sentiment Labeling
===========================

Chosen Approach:
----------------
Used pre-trained RoBERTa model (`cardiffnlp/twitter-roberta-base-sentiment`)
- Optimized for social media/informal text like employee messages

Output Generation:
-------------------
For each message, generated:
- label: Sentiment class (Positive, Negative, Neutral)
- roberta_pos: Confidence score for positive
- roberta_neg: Confidence score for negative
- roberta_neu: Confidence score for neutral

Data Augmentation:
-------------------
Added 3 new columns to the dataset:
- label
- roberta_pos
- roberta_neg
- roberta_neu


===========================
Task 2: Exploratory Data Analysis (EDA)
===========================

Process & Key Findings:
------------------------
- Dataset was clean, no major missing values
- Ready for analysis without extra cleaning

Sentiment Distribution:
------------------------
- Mostly Neutral messages
- Followed by Positive
- Few but notable Negative messages

Trends Over Time:
------------------
- Fluctuating sentiment patterns (no steady increase/decrease)
- Periodic dips suggest recurring stress points
- Likely aligned with business cycles


===========================
Task 3: Employee Score Calculation
===========================

Scoring Method:
----------------
- Sentiment labels mapped to numeric scores
- Grouped by month and sender
- Monthly sentiment score = sum of scores per employee


===========================
Task 4: Employee Ranking
===========================

Top 3 Senders with the Lowest Sentiment Scores:
------------------------------------------------
Month      | Sender                          | Score
-----------|----------------------------------|-------
2010-01    | sally.beck@enron.com            |   -1
2010-01    | bobette.riner@ipgdirect.com     |    0
2010-01    | john.arnold@enron.com           |    0

Top 3 Senders with the Highest Sentiment Scores:
-------------------------------------------------
Month      | Sender                          | Score
-----------|----------------------------------|-------
2010-03    | sally.beck@enron.com            |    6
2010-04    | kayne.coulter@enron.com         |    5
2010-02    | bobette.riner@ipgdirect.com     |    4


===============================
Task 5: Flight Risk Identification
===============================

Method:
--------
- Used a 30-day rolling window to count negative messages per employee
- Flagged employees with ≥4 negative messages in any window

Flight Risk Employees:
------------------------
- bobette.riner@ipgdirect.com
- john.arnold@enron.com
- don.baughman@enron.com
- sally.beck@enron.com
- rhonda.denton@enron.com


===========================
Task 6: Sentiment Forecasting
===========================

Approach 1: Linear Regression + LassoCV
----------------------------------------
1. Feature Selection:
   - Used LassoCV to select top 5 features:
     rolling_mean_3d, rolling_mean_7d,
     rolling_kurtosis_10d, rolling_skew_7d,
     rolling_skew_10d

2. Missing Values:
   - Filled using mean imputation

3. Model Training:
   - 80/20 train-test split
   - Linear Regression model

4. Forecast Simulation:
   - Simulated 60 future days
   - Sampled past 30 days (with replacement)

5. Evaluation:
   - MSE: 0.1139
   - R² Score: 0.3802

6. Visualization:
   - Weekly aggregation plot (actual vs forecast)

Approach 2: Prophet + LassoCV Regressors
-----------------------------------------
1. Data Prep:
   - Aggregated sentiment_score daily
   - Filled missing days using linear interpolation

2. Model Setup:
   - Used selected Lasso features as regressors
   - Training set: all data except last 30 days
   - Forecasted for last 30 days

3. Evaluation:
   - MSE: 0.0362
   - MAE: 0.1561

4. Visualization:
   - Forecast closely followed actual values
   - Detected weekly seasonality and upward trend

Final Verdict:
---------------
- Prophet + Lasso = Best for short-term accuracy, interpretability
- Linear Regression = Simpler, customizable for long-term scenarios
