Setup:

installing required libraries
pip install numpy pandas matplotlib seaborn transformers scipy torch torchvision torchaudio huggingface_hub[hf_xet]

Usage:

we will use numpy for translating tensors
pandas for data handling/preprocessing
matplotlib and seaborn for visualisation 
scipy for softmax function for smoothing out values obtained

Methodology:

we will use AutoTokenizer and AutoModelForSequenceClassification from transformers
AutoTokenizer--> for tokenising the text
AutoModelForSequenceClassification--> for model selection

then we will select the pretrained model like in this case :
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
then we will use tokenizer and hand the encoded text to model for sentiment analysis
result obtain from the model will be inform of tensors 
then we will convert tensors in numpy and smooth out values using softmax function 

We will apply this to all of our dataset!!!



