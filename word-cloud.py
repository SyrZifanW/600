from wordcloud import WordCloud
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt

def alpha_filter(w):
  # pattern to match a word of non-alphabetical characters
    pattern = re.compile('^[^a-z]+$')
    if (pattern.match(w)):
        return True
    else:
        return False

def pre_process(r):                               #processing the comments, turn them into the format that can be analyzed
  r = nltk.word_tokenize(r)                           #tokenization
  r = [w.lower() for w in r]                          #turn them into lower case
  r = [word for word in r if not any(c.isdigit() for c in word)]            #delete all the numbers
  stop = nltk.corpus.stopwords.words('english')                     #delete all the stop words
  stop.append("virus")
  stop.append("coronavirus")
  stop.append("corona")
  stop.append("n't")
  stop.append("corona")
  r = [w for w in r if not alpha_filter(w)]
  r = [x for x in r if x not in stop]
  r = [y for y in r if len(y) > 2]
  wnl = nltk.WordNetLemmatizer()                          #lemmatizing
  r= [wnl.lemmatize(t) for t in r]
  text = " ".join(r)
  return text


def remove_at(text, lang):
    result = []
    for i in range(len(text)):
        if lang[i] != 'en':
            continue
        tweet = text[i]
        tweet = re.sub(r'@\w*', '', tweet)
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        tweet = re.sub(r'https?://\S+[\r\n\s]*', '', tweet)
        tweet = re.sub(r'#', '', tweet)
        tweet = re.sub(r'&amp;', '', tweet)
        result.append(tweet)
        # word = text[i].split(' ')
        # print('1')
        # for j in word:
        #     if('@' in j) or ('#' in j) or ("http" in j) or ('&' in j):
        #         word.remove(j)
        # a = " ".join(word)
        # result.append(a)
    return result

f0 = pd.read_csv("2020-03-12 Coronavirus Tweets.CSV")
f0_text = f0['text']
f0_lang = f0['lang']

f = open("text.txt",'w',encoding = 'utf-8')

text = f0_text.tolist()
lang = f0_lang.tolist()
train_text = remove_at(text, lang)

for i in range(len(train_text)):
    train_text[i] = pre_process(train_text[i])

for i in train_text:
    f.write(i)

f.close()

f2 = open("text.txt",'r',encoding='utf-8').read()

wordcloud = WordCloud(
        background_color="white", #设置背景为白色，默认为黑色
        width=1500,              #设置图片的宽度
        height=960,              #设置图片的高度
        margin=10               #设置图片的边缘
        ).generate(f2)

plt.imshow(wordcloud)
# 消除坐标轴
plt.axis("off")
# 展示图片
plt.show()
# 保存图片
wordcloud.to_file('before 13.png')
plt.close()








