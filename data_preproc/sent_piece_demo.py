import sentencepiece as spm
from prepro_utils import preprocess_text, encode_ids
import os
# some code omitted here...
# initialize FLAGS

wkdir = os.path.dirname(os.path.abspath(''))
spiece_model_file = os.path.join(wkdir, 'model_cache/xlnet_cased_L-12_H-768_A-12/spiece.model')

text = "An input text string. pan viva build ElasticSearch to host netgear v7610 documents"

sp_model = spm.SentencePieceProcessor()
sp_model.Load(spiece_model_file)
text2 = preprocess_text(text, lower=False)
ids = encode_ids(sp_model, text2)

cc = 0
