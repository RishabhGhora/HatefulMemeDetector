import sys
from os import walk
import pandas as pd

from PIL import Image
import numpy as np
from OCR import get_text
from text_only_model import Predictor as TextOnlyPredictor
from transformers_model import Predictor as TransformersPredictor

# total arguments
n = len(sys.argv)

# file names 
filenames = []

# directory bool
isDirectory = False
directory = None

for i in range(1, n):
    if ("." not in sys.argv[i]):
      # directory
      filenames = next(walk(sys.argv[i]), (None, None, []))[2]  # [] if no file
      isDirectory = True
      directory = sys.argv[i]
      break
    filenames.append(sys.argv[i])

# Load models
transformers_predictor = TransformersPredictor()
text_only_predictor = TextOnlyPredictor()

# Lists for pandas df 
detected_text, transformer_prediction, transformer_prob, text_only_prediction, text_only_prob = [], [], [], [], []

for filename in filenames:
  if isDirectory:
    filename = directory+'/'+filename
  im = Image.open(filename)
  arr = np.asarray(im)
  text = get_text(arr)
  classification, transformers_probs = transformers_predictor.evaluate(filename,text)
  txt_classification, text_only_probs = text_only_predictor.evaluate(text)
  detected_text.append(text)
  transformer_prediction.append(classification)
  transformer_prob.append(transformers_probs)
  text_only_prediction.append(txt_classification)
  text_only_prob.append(text_only_probs)

df = pd.DataFrame(
    {'detected_text': detected_text,
     'transformer_prediction': transformer_prediction,
     'transformer_prob': transformer_prob,
     'text_only_prediction': text_only_prediction,
     'text_only_prob': text_only_prob
    })
df.to_csv("results.csv", sep=",")

