# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# !pip install MeCab

import MeCab

# +
text = '庭には二羽鶏がいる。'
mecab = MeCab.Tagger('mecabrc')
mecab.parse('')

node = mecab.parseToNode(text)
while node:
    print(node.surface, node.feature)
    node = node.next

# +
text = "私はお昼休みに美味しいチョコレートケーキを食べました"
mecab = MeCab.Tagger('mecabrc')
mecab.parse('')

node = mecab.parseToNode(text)
while node:
    print(node.surface, node.feature)
    node = node.next
# -

noun_list = []
verb_list = []
adjective_list = []

# +
node = mecab.parseToNode(text)

while node:
    if node.feature.split(",")[0] == "名詞":
        noun_list.append(node.surface)
    elif node.feature.split(",")[0] == "動詞":
        verb_list.append(node.feature.split(",")[6])
    elif node.feature.split(",")[0] == "形容詞":
        adjective_list.append(node.feature.split(",")[6])
        
    node = node.next
# -

noun_list

verb_list

adjective_list

# +
node = mecab.parseToNode(text)

words_list = []
while node:
    if node.feature.split(",")[0] == "動詞":
        words_list.append(node.feature.split(",")[6])
    elif node.feature.split(",")[0] == "形容詞":
        words_list.append(node.feature.split(",")[6])
    else:
        words_list.append(node.surface)
        
    node = node.next
# -

words_list[1:-1]


def mecab_sep(text):
    node = mecab.parseToNode(text)

    words_list = []
    while node:
        if node.feature.split(",")[0] == "動詞":
            words_list.append(node.feature.split(",")[6])
        elif node.feature.split(",")[0] == "形容詞":
            words_list.append(node.feature.split(",")[6])
        else:
            words_list.append(node.surface)

        node = node.next
        
    return words_list[1:-1]


mecab_sep("タガタメは3年経っても奥が深いなぁ")

# ## Bug of words / TF-IDF 
# https://www.youtube.com/watch?v=5vRyPMBOr_w

from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF 両方効かせる
vectorizer = TfidfVectorizer(analyzer=mecab_sep)
vecs = vectorizer.fit_transform(["こんにちは、これからよろしくお願いします", "こんばんは、これからよろしくお願いします"])
vecs.toarray()

# TF は効かせるが、IDFは効かせない　：　頻出ワードの抽出向け
vectorizer = TfidfVectorizer(analyzer=mecab_sep, use_idf=False)
vecs = vectorizer.fit_transform(["こんにちは、こんにちは、これからよろしくお願いします", "こんばんは、これからよろしくお願いします"])
vecs.toarray()

# 重複を除く場合
vectorizer = TfidfVectorizer(analyzer=mecab_sep, binary=True,  use_idf=False)
vecs = vectorizer.fit_transform(["こんにちは、こんにちは、これからよろしくお願いします", "こんばんは、これからよろしくお願いします"])
vecs.toarray()


def calc_vecs(docs):
    vectorizer = TfidfVectorizer(analyzer=mecab_sep)
    vecs = vectorizer.fit_transform(docs)
    return vecs.toarray()


## コサイン類似度
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity([[1, 2, 3]], [[3, 4, 5], [4, 3, 2], [1, 6, 3]])

target_docs_df = pd.read_csv("target_docs.csv")
target_docs_df

target_docs_df["文章リスト"].tolist()

# +
target_docs = target_docs_df["文章リスト"].tolist()
input_doc = "私は犬がすごく好きです。"

all_docs = [input_doc] + target_docs
all_docs_vecs = calc_vecs(all_docs)
all_docs_vecs
# -

cosine_similarity([all_docs_vecs[0]], all_docs_vecs[1:])

similarity = cosine_similarity([all_docs_vecs[0]], all_docs_vecs[1:])
target_docs_df["類似度"] = similarity[0]
target_docs_df.sort_values("類似度", ascending=False)

# ## ワードクラウド

# ! pip install wordcloud

# +
from wordcloud import WordCloud
 
# テキストファイル読み込み
text = open("constitution.txt", encoding="utf8").read()
 
# 画像作成
wordcloud = WordCloud(max_font_size=40).generate(text)
 
# 画像保存
wordcloud.to_file("result.png")
# -

from IPython.display import Image
Image("./result.png")

# +
from wordcloud import WordCloud
 
FONT_PATH = "ipaexg.ttf"
TXT_NAME = "rasyomon"

def get_word_str(text):
    import MeCab
    import re
 
    mecab = MeCab.Tagger()
    parsed = mecab.parse(text)
    lines = parsed.split('\n')
    lines = lines[0:-2]
    word_list = []
 
    for line in lines:
        tmp = re.split('\t|,', line)
 
        # 名詞のみ対象
        if tmp[1] in ["名詞"]:
            # さらに絞り込み
            if tmp[2] in ["一般", "固有名詞"]:
                word_list.append(tmp[0])
 
    return " " . join(word_list)
 
# テキストファイル読み込み
read_text = open(TXT_NAME + ".txt", encoding="utf8").read()
 
# 文字列取得
word_str = get_word_str(read_text)
 
# 画像作成
wc = WordCloud(font_path=FONT_PATH, max_font_size=40).generate(word_str)
 
# 画像保存（テキストファイル名で）
wc.to_file(TXT_NAME + ".png")
# -

from IPython.display import Image
Image(TXT_NAME + ".png")

# ## 結果をマスクに入れる

# +
from PIL import Image
import numpy as np

img = Image.open("./test.jpg")
mask_img = np.array(img)
# -

img

# +
# 画像作成
wc = WordCloud(background_color="white", mask=mask_img, contour_color='steelblue', font_path=FONT_PATH).generate(word_str)
 
# 画像保存（テキストファイル名で）
wc.to_file(TXT_NAME + "_with_mask.png")
# -

from IPython.display import Image
Image(TXT_NAME + "_with_mask.png")
