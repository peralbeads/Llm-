import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm
plt.style.use('ggplot')

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

df = pd.read_csv("test.csv")

example = df['body'][0]
print(example)


def polarity_score(eg):
    encoded_text = tokenizer(eg, return_tensors='pt')
    output = model(**encoded_text)
    score = output[0][0].detach().numpy()
    score = softmax(score)

    score_dict = {
        'roberta_neg': score[0],
        'roberta_neu': score[1],
        'roberta_pos': score[2],
        'label': ['negative', 'neutral', 'positive'][np.argmax(score)]
    }
    return score_dict


res = []

for i, rows in tqdm(df.iterrows(), total=len(df)):
    body = rows['body']
    roberta_result = polarity_score(body)
    res.append(roberta_result)


result_df = pd.DataFrame(res)

print(result_df.head(10))
result_df = pd.concat([df, result_df], axis=1)

# Save results
result_df.to_csv("roberta_sentiment_analysis.csv", index=False)


fig, axs = plt.subplots(1, 3, figsize=(15, 5))

sns.histplot(data=result_df, x='roberta_neg', ax=axs[0])
sns.histplot(data=result_df, x='roberta_neu', ax=axs[1])
sns.histplot(data=result_df, x='roberta_pos', ax=axs[2])

axs[0].set_title("Negative")
axs[1].set_title("Neutral")
axs[2].set_title("Positive")


plt.tight_layout()
plt.show()

