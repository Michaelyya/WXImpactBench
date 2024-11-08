import re
import pandas as pd
text = '''!rPFTilfnir e ," 🌍🌍.. ,, i w -Lii tra i?n Finn ft3 It M t f 1 IM 3 r lie sview INSIDE Editorials Macpherson B3 1 1 S Water, water Is Canada quietly preparing the plumbing for a $100-billion plan. to sell northern fresh water to the Prairies. the United States'''

def truncate(text):
  """truncate to second last sentence if incomplete"""
  #split text into sentences
  sentences = re.split(r"(?<=[.!?]) +", text)
  if len(sentences) < 2:
      return text
    #if last sentence ends properly
  last_sent = sentences[-1].strip()
  if last_sent and last_sent[-1] not in [".", "!", "?"]:
      return " ".join(sentences[:-1]).strip()
  return text.strip()

def clean_text(text):
    # replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()
    #replace strikes *
    text = re.sub(".[*]", "", text)
    # OCR's common mistakes on isolated character
    text = re.sub(r"\b0\b", "O", text)  # single "0" -> "O"
    text = re.sub(r"\b1\b", "I", text)  # single "1" -> "I"
    text = re.sub(r"\b5\b", "S", text)  # single "5" -> "S"
    text = re.sub(r'([.,;:!?\'"`])\1+', r"\1", text)  # remove isolated repeated punctuation and only keep 1
    # remove non-ASCII char
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    #remove any non-word character at str beginning
    text = re.sub("^\W", "", text)
    #remove double comma
    text = re.sub(",{2,3}", ",", text)
    #remove invalid unicode
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

csv_path="selected_query.csv"
df=pd.read_csv(csv_path).dropna(subset=["Text"])
# remove the dates from micheal's entries of texts
df.loc[df["SelectionBy"] == "Michael", "Text"] = df.loc[df["SelectionBy"] == "Michael", "Text"].str.replace(r'^.*?,', "", regex=True)

df["Text_Length"] = df["Text"].apply(len)
df["Word_Count"] = df["Text"].apply(lambda text: len(str(text).split()))
df["cleaned_text"] = df["Text"].apply(clean_text)
df.to_csv("cleaned_selected_query.csv", index=False)