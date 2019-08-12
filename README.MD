
![project_로이](https://user-images.githubusercontent.com/50260643/62846598-5bdd5580-bd0c-11e9-983d-f10e9ee93f8a.png)


![concept](https://user-images.githubusercontent.com/50260643/62846679-f76ec600-bd0c-11e9-900c-591be7d4b5ae.png)

![model](https://user-images.githubusercontent.com/50260643/62846714-2c7b1880-bd0d-11e9-9c67-3ca5b18c1e46.png)


![로이_설명](https://user-images.githubusercontent.com/50260643/62846728-49175080-bd0d-11e9-9fba-6fc99335507e.png)

### [보유 기술 및 개발 환경]
- Python, Tensorflow, Gensim, Word2Vec, Konlpy(Komoran), NLTK, Pandas, Seq2Seq
- Pycharm, Jupyter, Windows, Oracle, Atom, Embedding Projector


### [참여 인원 및 본인 기여도]
- 참여 인원 3명

- 구분 파트
	- 웹 크롤링 
	- 데이터 전처리 및 Word2Vec
	- seq2Seq, DB 부분으로 구분 중

- 모델링 및 데이터 전처리 및 Word2Vec 파트를 담당(기여도 50%)



#### 첫번째 테스트 케이스
- '교통사고'키워드로 검색한 판례 375개를 Word2Vec 모델의 데이터 셋으로 활용한 테스트 케이스1.

###### 두번째 테스트 케이스(2271개 판례 활용)
- 5개의 카테고리(교통사고, 교통, 사고, 운전, 차량)를 사용하여 Word2Vec 모델의 데이터 셋으로 활용한 테스트 케이스2.



이 코드는 첫번째 테스트 케이스의 내용으로 구성하였습니다.

#### Summary
데이터 전처리 작업 및 사용자의 입력값(Ex. 도로에서 음주 후 음주운전 중에 사고가 났고, 당시 알콜 농도는 0.15였습니다)을 토크나이징하여 단어값들과 가장 연관성이 높은 Word2Vec모델의 값을 이용할 예정이며, 그 값이 높을수록 해당 판례와 관련성이 많다고 가정하였습니다. 때문에 각 판례들을 각각 파싱하여 단어 구성과 단어의 갯수를 딕셔너리로 구성합니다.
"1차 완료"부분

'교통사고'판례에서 나온 단어들을 토크나이징하고 그 중 사용할 단어와 사용하지 않을 단어 셋을 구성하였고,사용하지 않을 단어를 제외한 Word2Vec모델에 넣을 단어 말뭉치를 구성하였습니다
"2차 완료"부분

```python
from gensim.models import Word2Vec
import numpy as np
from pprint import pprint
from konlpy.tag import Komoran;
k = Komoran()  
import nltk  
from nltk import FreqDist 
import pandas as pd
import codecs
import pprint
```


```python
# 1차적으로 사용하지 않을 키워드, 판례에서 필요한 부분(판례 제목, 사건 번호 등을 따로 뽑아 놓음)
fileName_dnusing_wordSet = 'C:/Users/Jongil Park/PycharmProjects/ai_study/New_projectDirectory_law2' \
                           '/DataSet/Gyotongsago_data_375/Gyotongsago_dataSet/Gyotongsago_dnUsingWordSet_375_onlyNoun.txt'
fileName_title = 'C:/Users/Jongil Park/PycharmProjects/ai_study/New_projectDirectory_law2' \
                 '/DataSet/Gyotongsago_data_375/Gyotongsago_dataSet/Gyotongsago_lawTitle_375.txt'
fileName_keyNum = 'C:/Users/Jongil Park/PycharmProjects/ai_study/New_projectDirectory_law2' \
                  '/DataSet/Gyotongsago_data_375/Gyotongsago_dataSet/Gyotongsago_lawNumber_375.txt'

## 불러온 파일 리스트화
with open(fileName_dnusing_wordSet, 'r') as infile:
    word_list = [line.rstrip() for line in infile]

## 타이틀만 모아놓은 것 리스트로 만들기
with open(fileName_title, 'r') as infile2:
    title_list = [line.rstrip() for line in infile2]

## 키워드 넘버만 모아놓은 것 리스트 만들기
try:
    with open(fileName_keyNum) as infile3:
        keyNum_list = [line.rstrip() for line in infile3]
except UnicodeDecodeError:   ## TODO: 텍스트 파일을 불러올 때, 에러가 나는 이유는 무엇인가? 왜 그래서 codecs를 써야 하는가?
    with codecs.open(fileName_keyNum, "r", "utf-8") as infile3:
        keyNum_list = [line.rstrip() for line in infile3]
```


```python
df_list = pd.DataFrame()
val_word_list = pd.Series(word_list)
val_title_list = pd.Series(title_list)
val_keyNum_list = pd.Series(keyNum_list)

df_list['단어'] = val_word_list
df_list['판례이름'] = val_title_list
df_list['사건번호'] = val_keyNum_list
```


```python
df_list.iloc[:20]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>단어</th>
      <th>판례이름</th>
      <th>사건번호</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>것</td>
      <td>특정범죄가중처벌등에관한법률위반(도주차량)·도로교통법위반(사고후미조치)</td>
      <td>2013도15885</td>
    </tr>
    <tr>
      <th>1</th>
      <td>위</td>
      <td>교통사고처리특례법위반</td>
      <td>2012도11431</td>
    </tr>
    <tr>
      <th>2</th>
      <td>수</td>
      <td>교통사고처리특례법위반</td>
      <td>2014도3235</td>
    </tr>
    <tr>
      <th>3</th>
      <td>소외</td>
      <td>교통사고처리특례법위반</td>
      <td>2015도3107</td>
    </tr>
    <tr>
      <th>4</th>
      <td>상고</td>
      <td>교통사고처리특례법위반·도로교통법위반</td>
      <td>2013도10958</td>
    </tr>
    <tr>
      <th>5</th>
      <td>등</td>
      <td>교통사고처리특례법위반·도로교통법위반(음주운전)·도로교통법위반(무면허운전)</td>
      <td>2013도15031</td>
    </tr>
    <tr>
      <th>6</th>
      <td>이</td>
      <td>교통사고처리특례법위반·도로교통법위반(무면허운전)·도로교통법위반·자동차손해배상보장법위반</td>
      <td>2015도686</td>
    </tr>
    <tr>
      <th>7</th>
      <td>점</td>
      <td>도로교통법위반(사고후미조치)·도로교통법위반(음주운전)</td>
      <td>2015도12451</td>
    </tr>
    <tr>
      <th>8</th>
      <td>항</td>
      <td>교통사고처리특례법위반</td>
      <td>2014노3022</td>
    </tr>
    <tr>
      <th>9</th>
      <td>공</td>
      <td>교통사고처리특례법위반</td>
      <td>2016도18941</td>
    </tr>
    <tr>
      <th>10</th>
      <td>중</td>
      <td>교통사고처리특례법위반</td>
      <td>2016도17442</td>
    </tr>
    <tr>
      <th>11</th>
      <td>부분</td>
      <td>특정범죄가중처벌등에관한법률위반(도주차량)·상해·공무집행방해·도로교통법위반(사고후미조...</td>
      <td>2016도12407</td>
    </tr>
    <tr>
      <th>12</th>
      <td>기재</td>
      <td>살인(예비적죄명:교통사고처리특례법위반)·사기(남편이 보험금을 노리고 교통사고를 내어...</td>
      <td>2017도1549</td>
    </tr>
    <tr>
      <th>13</th>
      <td>심</td>
      <td>유기치사·교통사고처리특례법위반(치사)</td>
      <td>2017고합146</td>
    </tr>
    <tr>
      <th>14</th>
      <td>규정</td>
      <td>특정경제범죄가중처벌등에관한법률위반(사기)·사기·폭력행위등처벌에관한법률위반(공동상해)...</td>
      <td>2016도21075</td>
    </tr>
    <tr>
      <th>15</th>
      <td>처리</td>
      <td>특정범죄가중처벌등에관한법률위반(도주차량)·교통사고처리특례법위반·도로교통법위반·도로교...</td>
      <td>2006노2898</td>
    </tr>
    <tr>
      <th>16</th>
      <td>경</td>
      <td>교통사고처리특례법위반</td>
      <td>2016노186</td>
    </tr>
    <tr>
      <th>17</th>
      <td>행위</td>
      <td>교통사고처리특례법위반</td>
      <td>2015고정263</td>
    </tr>
    <tr>
      <th>18</th>
      <td>후</td>
      <td>교통사고처리특례법위반</td>
      <td>2015노3482</td>
    </tr>
    <tr>
      <th>19</th>
      <td>주의의무</td>
      <td>교통사고처리특례법위반</td>
      <td>88도2010</td>
    </tr>
  </tbody>
</table>
</div>




```python
## 여기서 pasing 된 corpus 값을 명사만 추출하여 각 단어별 갯수를 딕셔너리로 만들어주고 append
def append_noun_words(corpus):
    noun_words = ['NNG', 'NNB', 'NP']  # 일반명사, 고유명사, 대명사만 학습  // 이렇게 한 이유에 대해서?
    results = []
    for text in corpus:
        for noun_word in noun_words:
            if noun_word in text[1]:
                results.append(text[0])
    return results
```


```python
## 여기서 pasing 된 corpus 값을 명사만 추출하여 각 단어별 갯수를 딕셔너리로 만들어주고 append
## {판례의 명사 키워드의 단어:갯수 매칭된 리스트}
total_panrye_parsingReason = []  # 이 안의 값은 딕셔너리가 되어야 함

## '교통사고' 하나의 카테고리만 활용할 때,
file_name = ['C:/Users/Jongil Park/PycharmProjects/ai_study/New_projectDirectory_law2/DataSet/Gyotongsago_data_375'
             '/Gyotongsago_dataSet/Gyotongsago_reason_375/({0}).txt']
file_len = [375]

for z in range(len(file_name)):

    for i in range(file_len[z]):    # len(title_list)
        ione_panrye_parsingReason = {}

        try:
            with open(file_name[z].format(i+1), 'r') as f:
                texts = f.read()
                corpus = k.pos("\n".join([s for s in texts.split("\n") if s]))
        except UnicodeDecodeError:
            with codecs.open(file_name[z].format(i+1), "r", "utf-8") as f:
                texts = f.read()
                corpus = k.pos("\n".join([s for s in texts.split("\n") if s]))

        corpus = append_noun_words(corpus)
        corpus_ko = nltk.Text(corpus, name="각 판례별 명사 처리")
        corpus_ko_vocab = corpus_ko.vocab()
        corpus_vocab_common = corpus_ko_vocab.most_common(len(corpus_ko_vocab))

    ## 여기에 필요한 키워드 값만 추출하는 것을 추가 할 것.
        for j in range(len(corpus_vocab_common)):
            if corpus_vocab_common[j][0] not in word_list:
                ione_panrye_parsingReason[corpus_vocab_common[j][0]] = corpus_vocab_common[j][1]
        total_panrye_parsingReason.append(ione_panrye_parsingReason)

print("1차 완료: 각 타이틀, 사건번호, 딕셔너리(단어: 번호)매칭")
```

    1차 완료: 각 타이틀, 사건번호, 딕셔너리(단어: 번호)매칭
    


```python
pprint.pprint(total_panrye_parsingReason)[:2]
```

    [{'가중': 1,
      '경우': 1,
      '경위': 3,
      '경찰': 4,
      '고의': 2,
      '관여': 1,
      '교통': 4,
      .
      .
      .
      .
      
      '피고인': 2,
      '피해자': 10,
      '한계': 1,
      '해석': 1,
      '허용': 1,
      '형사소송법': 1,
      '회복': 1}]

```python
#======================================================================================================
## TODO: 3번: W2V을 위한 데이터 전처리, '기준'을 어떻게 둘 것인가?
"""최종적으로 데이터 프레임화를 위한 리스트"""
# TODO: 판례 데이터 사용방법 
# 1. '교통사고'의 이유 부분 전체 합을 이용
# 2. '교통사고' 각 판례(375개)를 각각 W2V 모델에 사용하고 그 값들을 활용?

# 각 교통사고 관련 카테고리의 '이유' 부분의 sum 값을 활용
data_file = ['C:/Users/Jongil Park/PycharmProjects/ai_study/New_projectDirectory_law2/DataSet/Gyotongsago_data_375'
             '/Gyotongsago_dataSet/Gyotongsago_totalSum_375.txt']

total_sumWord = []
for i in range(len(data_file)):
    try:
        with open(data_file[i]) as f:
            texts = f.read()
            corpus = k.pos("\n".join([s for s in texts.split("\n") if s]))  # 이것의 의미 다시 한번 생각

    except UnicodeDecodeError:
        with codecs.open(data_file[i], "r", "utf-8") as f:
            texts = f.read()
            corpus = k.pos("\n".join([s for s in texts.split("\n") if s]))

    corpus_noun = append_noun_words(corpus)
    corpus_ko = nltk.Text(corpus_noun, name="각 판례별 명사 처리")
    corpus_ko_vocab = corpus_ko.vocab()
    corpus_ko_vocab_items = corpus_ko_vocab.items()
    corpus_ko_vocab_items_list = list(corpus_ko_vocab_items)
    #corpus_vocab_common = corpus_ko_vocab.most_common(len(corpus_ko))

    ## new_corpus 는 살릴 단어가 하나씩 밖에 안 들어가 있다. 그 개수가 빠져있는 합.
    new_corpus = []
    corpus_values = []
    new_corpus2 = []

    for z in range(len(corpus_ko_vocab_items_list)):
        corpus_word = corpus_ko_vocab_items_list[z][0]
        if corpus_word not in word_list:
            new_corpus.append([corpus_word])
            corpus_values.append(corpus_ko_vocab_items_list[z][1])
    new_corpus2 = sum(new_corpus, [])

    ## 이것이 corpus 에서 쓰지 않은 키워드를 갯수만큼 제외시키는 것
    word_corpus_last = [x for x in corpus_noun if x not in word_list]
    word_corpus_last2 = []
    for z in range(len(word_corpus_last)):
        word_corpus_last2.append([word_corpus_last[z]])   # word_corpus_last2.append(word_corpus_last[z].split())에서 변경

    total_sumWord.append(word_corpus_last2)
    # total_sumWord.append(new_corpus)
total_sumWord = sum(total_sumWord, [])
df4 = pd.DataFrame(total_sumWord)
df4.to_excel('C:/Users/Jongil Park/PycharmProjects/ai_study/New_projectDirectory_law2/DataSet/Gyotongsago_data_375/'
             'Gyotongsago_resultSet/total_sumWord_for_W2V.xlsx',
             sheet_name='최종W2V단어말뭉치')

print("word2vec 전, 2차 완료(Word2Vec모델에 넣을 단어 말뭉치 형성)")
print("len(total_sumWord): ", len(total_sumWord))
```

    word2vec 전, 2차 완료(Word2Vec모델에 넣을 단어 말뭉치 형성)
    len(total_sumWord):  63352
    


```python
df_corpus = pd.DataFrame()
val_corpus = pd.Series(corpus)
val_corpus_noun = pd.Series(corpus_noun)
val_corpus_ko_vocab_items_list = pd.Series(corpus_ko_vocab_items_list)

df_corpus['corpus'] = val_corpus
df_corpus['corpus_noun'] = val_corpus_noun
df_corpus['corpus_items'] = val_corpus_ko_vocab_items_list
```


```python
df_corpus.iloc[:20]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>corpus</th>
      <th>corpus_noun</th>
      <th>corpus_items</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(﻿, SW)</td>
      <td>원심</td>
      <td>(원심, 1628)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(원심, NNP)</td>
      <td>판결</td>
      <td>(판결, 1422)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(판결, NNG)</td>
      <td>파기</td>
      <td>(파기, 301)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(을, JKO)</td>
      <td>사건</td>
      <td>(사건, 1954)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(파기, NNG)</td>
      <td>대전지방법원</td>
      <td>(대전지방법원, 13)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(하, XSV)</td>
      <td>본원</td>
      <td>(본원, 22)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(고, EC)</td>
      <td>합의</td>
      <td>(합의, 125)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(,, SP)</td>
      <td>부</td>
      <td>(부, 114)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(사건, NNG)</td>
      <td>환송</td>
      <td>(환송, 169)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(을, JKO)</td>
      <td>상고</td>
      <td>(상고, 703)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(대전지방법원, NNP)</td>
      <td>이유</td>
      <td>(이유, 1086)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(본원, NNP)</td>
      <td>판단</td>
      <td>(판단, 714)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(합의, NNG)</td>
      <td>특정범죄 가중처벌 등에 관한 법률</td>
      <td>(특정범죄 가중처벌 등에 관한 법률, 57)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(부, NNG)</td>
      <td>조의</td>
      <td>(조의, 57)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(에, JKB)</td>
      <td>도주</td>
      <td>(도주, 152)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(환송, NNG)</td>
      <td>차량</td>
      <td>(차량, 1438)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>(하, XSV)</td>
      <td>운전자</td>
      <td>(운전자, 426)</td>
    </tr>
    <tr>
      <th>17</th>
      <td>(ㄴ다, EF)</td>
      <td>가중</td>
      <td>(가중, 58)</td>
    </tr>
    <tr>
      <th>18</th>
      <td>(., SF)</td>
      <td>처벌</td>
      <td>(처벌, 291)</td>
    </tr>
    <tr>
      <th>19</th>
      <td>(상고, NNP)</td>
      <td>규정</td>
      <td>(규정, 619)</td>
    </tr>
  </tbody>
</table>
</div>




```python
# new_corpus2 - 키워드 단어 제외하되, 중복값을 허용X, len(new_corpus2) = 948
# 단, 단어 중복을 허용하지 않고 1개만 남기기 때문에, 본래 데이터의 가치를 잃을 수 있다.

# word_corpus_last - 키워드 단어 제외하되, 중복값 허용, len(word_corpus_last) = 63352 
# total_sumWord - Word2Vec에 넣기 전 최종 단어 set, len(total_sumWord) = 63352 (375개 set일 때 같지만, 2271개일 때 269333개)
df_corpus_last = pd.DataFrame()
val_new_corpus2 = pd.Series(new_corpus2)
val_corpus_values = pd.Series(corpus_values)
val_word_corpus_last = pd.Series(word_corpus_last)
val_total_sumWord = pd.Series(total_sumWord)

df_corpus_last['new_corpus2'] = val_new_corpus2
df_corpus_last['corpus_values'] = val_corpus_values
df_corpus_last['word_corpus_last'] = val_word_corpus_last
df_corpus_last['최종 단어'] = val_total_sumWord 
```


```python
df_corpus_last.iloc[:20]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>new_corpus2</th>
      <th>corpus_values</th>
      <th>word_corpus_last</th>
      <th>최종 단어</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>원심</td>
      <td>1628</td>
      <td>원심</td>
      <td>[원심]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>판결</td>
      <td>1422</td>
      <td>판결</td>
      <td>[판결]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>사건</td>
      <td>1954</td>
      <td>사건</td>
      <td>[사건]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>이유</td>
      <td>1086</td>
      <td>이유</td>
      <td>[이유]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>판단</td>
      <td>714</td>
      <td>판단</td>
      <td>[판단]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>도주</td>
      <td>152</td>
      <td>도주</td>
      <td>[도주]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>차량</td>
      <td>1438</td>
      <td>차량</td>
      <td>[차량]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>운전자</td>
      <td>426</td>
      <td>운전자</td>
      <td>[운전자]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>가중</td>
      <td>58</td>
      <td>가중</td>
      <td>[가중]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>처벌</td>
      <td>291</td>
      <td>처벌</td>
      <td>[처벌]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>보호법익</td>
      <td>14</td>
      <td>보호법익</td>
      <td>[보호법익]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>사고</td>
      <td>1781</td>
      <td>사고</td>
      <td>[사고]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>경위</td>
      <td>109</td>
      <td>경위</td>
      <td>[경위]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>피해자</td>
      <td>1844</td>
      <td>피해자</td>
      <td>[피해자]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>상해</td>
      <td>218</td>
      <td>상해</td>
      <td>[상해]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>부위</td>
      <td>119</td>
      <td>부위</td>
      <td>[부위]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>정황</td>
      <td>34</td>
      <td>사고</td>
      <td>[사고]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>도로교통법</td>
      <td>511</td>
      <td>정황</td>
      <td>[정황]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>조치</td>
      <td>401</td>
      <td>사고</td>
      <td>[사고]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>인정</td>
      <td>948</td>
      <td>운전자</td>
      <td>[운전자]</td>
    </tr>
  </tbody>
</table>
</div>



First corpus data for Word2Vec
- 63352개의 단어 말뭉치 확보(각 단어의 중복 및 순서 혀용) = total_sumWord

Second corpus data for Word2Vec
- 948개의 단어 말뭉치 확보(각 단어 중복 허용 x) = new_corpus


![W2V_설명](https://user-images.githubusercontent.com/50260643/62867653-c31bf980-bd4e-11e9-9659-7ac0b155e37f.png)



### 첫번째, new_corpus 값을 활용


```python
## Word2Vec
import time
startTime = time.time()

word2vec_model = Word2Vec(new_corpus, size=200,     # 인자값에 new_corpus(948개) 혹은 total_sumWord(63352개)
                          window=2,
                          min_count=0,
                          workers=4,
                          iter=10000, sg=1)

w2v_key = word2vec_model.wv.vocab.keys()
input_keyword_last = ['테스트를 위한 빈값 넣어놓은 것']
print("len(w2v_key):", len(w2v_key))

endTime = time.time() - startTime
print(endTime)
```

    len(w2v_key): 948
    101.29475426673889
    

Word2Vec모델을 통해 나온 중복 제외 최종 키워드 갯수: 596


```python
def input_to_keyword():
    input_text = input("검색어를 입력하세요(끝내시려면 enter 키를 눌러주세요): ")
    k = Komoran()    # 새로 변수를 정의하지 않을시 반복해서 사용자가 검색어를 입력하면 에러가 뜨는 경우가 발생
    if input_text is '':
        input_keyword_last = None
        return input_keyword_last
    else:
        input_corpus = k.pos("\n".join([s for s in input_text.split("\n") if s]))
        parsed_input = append_noun_words(input_corpus)
        input_keyword_last = []
        for j in range(len(parsed_input)):
            # if parsed_input[j] not in word_list:
            # 위에 문장을 쓰지 않는 이유는 사용자가 검색한 키워드가 항상 우리 키워드 안에 있지는 않지만, 사용자의 키워드를 보여줄 필요는 있다.
            input_keyword_last.append(parsed_input[j])
        return input_keyword_last

repeat_num = 1    # 저장할 액셀 파일에 숫자를 붙여주기 위한 기초값, 첫번째를 뜻함.

while True:
    input_keyword_last = input_to_keyword()  # 인풋값을 사용할 키워드로 정리한 값
    if input_keyword_last is None:
        break
    columns_keyword = "·".join(input_keyword_last)  # 컬럼 이름을 위한 합치기

    df = pd.DataFrame()
    predict_words = []
    predict_values = []

    ## TODO: 과연 내가 찾으려고 하는 연관성이 predict_output_word()가 맞는가?
    predict_model = word2vec_model.predict_output_word(input_keyword_last, topn=len(w2v_key))
    for n in range(len(predict_model)):
        predict_words.append((predict_model[n][0]))
        predict_values.append(predict_model[n][1])
    val_words = pd.Series(predict_words)
    val_values = pd.Series(predict_values)
    df[columns_keyword] = val_words
    df['vector값'] = val_values
    #word2vec_model.max_final_vocab  # 이것은 왜 필요한거지? 아마 그냥 들어간거 같은데!!??

    df.to_csv('C:/Users/Jongil Park/PycharmProjects/ai_study/New_projectDirectory_law2/DataSet/Gyotongsago_data_375/Gyotongsago_resultSet'
              '/Gyotongsago_375_W2V_predict({0}).csv'.format(repeat_num),
              encoding='CP949')

    print("({0})차 구동완료".format(repeat_num))

    ## 키워드 연관 단어 상위 20개를 기준으로 한 카운트 값
    df_top20 = df[columns_keyword].iloc[:20]   ## iloc와 loc 차이? 여기에선 같은 값이 나오지만 둘의 차이 명확히 알기
    count_list_total = []
    for kk in range(len(total_panrye_parsingReason)):
        count = 0
        for j in range(len(df_top20)):
            if df_top20[j] in total_panrye_parsingReason[kk]:
                # 현재 count 값은 상위 20개를 순서상관없이 + 값, 순서별로 가중치를 주는 방법도 고려할 것.
                count += total_panrye_parsingReason[kk][df_top20[j]]
        count_list_total.append(count)

    # 카운트 값이 높은 순서의 인덱스 값의 리스트를 만들기
    df_count = pd.DataFrame()
    val_title_list = pd.Series(title_list)
    val_keyNum_list = pd.Series(keyNum_list)

    df_count = pd.DataFrame({(columns_keyword): count_list_total})
    df_count['판례이름'] = val_title_list
    df_count['사건번호'] = val_keyNum_list
    last_df_count = df_count.sort_values(by=columns_keyword, ascending=False)

    last_df_count.to_csv('C:/Users/Jongil Park/PycharmProjects/ai_study/New_projectDirectory_law2/DataSet'
                         '/Gyotongsago_data_375/Gyotongsago_resultSet/Gyotongsago_375_W2V_result({0}).csv'.format(repeat_num),
                         encoding='CP949')
    repeat_num += 1

print("구동완료")
```

    검색어를 입력하세요(끝내시려면 enter 키를 눌러주세요): 도로에서 술을 먹고 음주 운전 뒤 반대편 차량과 부딪쳤습니다.
    (1)차 구동완료
    검색어를 입력하세요(끝내시려면 enter 키를 눌러주세요): 
    구동완료
    


```python
df.iloc[:20]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>도로·술·음주·운전·뒤·반대편·차량</th>
      <th>vector값</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>원심</td>
      <td>0.001055</td>
    </tr>
    <tr>
      <th>1</th>
      <td>전조등</td>
      <td>0.001055</td>
    </tr>
    <tr>
      <th>2</th>
      <td>감속</td>
      <td>0.001055</td>
    </tr>
    <tr>
      <th>3</th>
      <td>소요</td>
      <td>0.001055</td>
    </tr>
    <tr>
      <th>4</th>
      <td>추산</td>
      <td>0.001055</td>
    </tr>
    <tr>
      <th>5</th>
      <td>후자</td>
      <td>0.001055</td>
    </tr>
    <tr>
      <th>6</th>
      <td>약국</td>
      <td>0.001055</td>
    </tr>
    <tr>
      <th>7</th>
      <td>설득력</td>
      <td>0.001055</td>
    </tr>
    <tr>
      <th>8</th>
      <td>추적</td>
      <td>0.001055</td>
    </tr>
    <tr>
      <th>9</th>
      <td>용기</td>
      <td>0.001055</td>
    </tr>
    <tr>
      <th>10</th>
      <td>수색</td>
      <td>0.001055</td>
    </tr>
    <tr>
      <th>11</th>
      <td>조회</td>
      <td>0.001055</td>
    </tr>
    <tr>
      <th>12</th>
      <td>통화</td>
      <td>0.001055</td>
    </tr>
    <tr>
      <th>13</th>
      <td>사망진단서</td>
      <td>0.001055</td>
    </tr>
    <tr>
      <th>14</th>
      <td>제보자</td>
      <td>0.001055</td>
    </tr>
    <tr>
      <th>15</th>
      <td>블랙박스</td>
      <td>0.001055</td>
    </tr>
    <tr>
      <th>16</th>
      <td>기로</td>
      <td>0.001055</td>
    </tr>
    <tr>
      <th>17</th>
      <td>배</td>
      <td>0.001055</td>
    </tr>
    <tr>
      <th>18</th>
      <td>가슴</td>
      <td>0.001055</td>
    </tr>
    <tr>
      <th>19</th>
      <td>구급차</td>
      <td>0.001055</td>
    </tr>
  </tbody>
</table>
</div>



예를 들어, 사용자의 입력이
"도로에서 술을 먹고 음주 운전 뒤 반대편 차량과 부딪쳤습니다."일 때,
해당 입력값을 파싱하여 Word2Vec모델에 넣고 가장 연관성이 높다고(높은 벡터값) 나온 단어 셋을 만들었습니다.


```python
last_df_count.iloc[:20]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>도로·술·음주·운전·뒤·반대편·차량</th>
      <th>판례이름</th>
      <th>사건번호</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>354</th>
      <td>93</td>
      <td>폭력행위등 처벌에 관한 법률위반(공동공갈)·업무 방해·공갈·폭력행위등 처벌에 관한 ...</td>
      <td>2011노163</td>
    </tr>
    <tr>
      <th>36</th>
      <td>46</td>
      <td>살인(예비적죄명:교통사고처리특례법위반)·사기</td>
      <td>2015노358</td>
    </tr>
    <tr>
      <th>290</th>
      <td>41</td>
      <td>특정범죄가중처벌등에관한법률위반(도주차량, 인정된 죄명 : 교통사고처리특례법위반)</td>
      <td>96도591</td>
    </tr>
    <tr>
      <th>12</th>
      <td>40</td>
      <td>살인(예비적죄명:교통사고처리특례법위반)·사기(남편이 보험금을 노리고 교통사고를 내어...</td>
      <td>2017도1549</td>
    </tr>
    <tr>
      <th>301</th>
      <td>34</td>
      <td>교통사고처리특례법위반</td>
      <td>93노4273</td>
    </tr>
    <tr>
      <th>221</th>
      <td>30</td>
      <td>교통사고처리특례법위반</td>
      <td>89도377</td>
    </tr>
    <tr>
      <th>293</th>
      <td>29</td>
      <td>교통사고처리특례법위반</td>
      <td>98도2605</td>
    </tr>
    <tr>
      <th>286</th>
      <td>28</td>
      <td>교통사고처리특례법위반·도로교통법위반</td>
      <td>96도1540</td>
    </tr>
    <tr>
      <th>281</th>
      <td>22</td>
      <td>살인·폭력행위등처벌에관한법률위반·특수공무집행방해치상·공용물건손상·도로교통법위반·향정...</td>
      <td>96도2588</td>
    </tr>
    <tr>
      <th>15</th>
      <td>19</td>
      <td>특정범죄가중처벌등에관한법률위반(도주차량)·교통사고처리특례법위반·도로교통법위반·도로교...</td>
      <td>2006노2898</td>
    </tr>
    <tr>
      <th>346</th>
      <td>17</td>
      <td>교통사고 처리 특례법 위반</td>
      <td>2011노938</td>
    </tr>
    <tr>
      <th>76</th>
      <td>17</td>
      <td>사기·교통사고처리특례법위반·유가증권위조(변경된죄명:유가증권변조)·위조유가증권행사(변...</td>
      <td>2005노396</td>
    </tr>
    <tr>
      <th>85</th>
      <td>17</td>
      <td>강도치상(인정된죄명:절도·상해)·특수강도·도로교통법위반(무면허운전)·사기·교통사고처...</td>
      <td>2007노193</td>
    </tr>
    <tr>
      <th>264</th>
      <td>17</td>
      <td>교통사고처리특례법위반</td>
      <td>94도1888</td>
    </tr>
    <tr>
      <th>253</th>
      <td>16</td>
      <td>교통사고처리특례법위반</td>
      <td>91도1746</td>
    </tr>
    <tr>
      <th>340</th>
      <td>16</td>
      <td>특정범죄가중처벌등에관한법률위반(영리약취·유인등)·아동·청소년의성보호에관한법률위반(강...</td>
      <td>2011노573</td>
    </tr>
    <tr>
      <th>90</th>
      <td>16</td>
      <td>특정범죄가중처벌등에관한법률위반(도주차량, 인정된 죄명 : 교통사고처리특례법위반)·도...</td>
      <td>99도5023</td>
    </tr>
    <tr>
      <th>280</th>
      <td>16</td>
      <td>교통사고처리특례법위반(추가된 죄명:도로법위반)</td>
      <td>96도2030</td>
    </tr>
    <tr>
      <th>262</th>
      <td>15</td>
      <td>교통사고처리특례법위반</td>
      <td>94도2393</td>
    </tr>
    <tr>
      <th>129</th>
      <td>15</td>
      <td>교통사고처리특례법위반ㆍ업무상과실자동차파괴</td>
      <td>83도3006</td>
    </tr>
  </tbody>
</table>
</div>



위에서 나온 벡터값이 높은 단어 상위 20개를 활용하여, 각 판례에 20개의 단어 들어있는 갯수대로 카운트를 매겼습니다.
위에 가정한대로 카운트 값이 높을수록 해당 판례가 가장 입력값과 연관성이 높은 판례라고 생각했습니다.
위에서 확보해놓은 각 판례의 {단어: 갯수}딕셔너리를 통하여 카운트 값이 가장 큰 값 순으로 정렬하였습니다.

DataFrame의 내용은
맨 왼쪽의 숫자는 1-375개의 판례의 순서 번호이며,
중간 숫자는 해당 키워드('도로·술·음주·운전·뒤·반대편·차량')의 카운트값
카운트값이 높은 순서대로 판례이름과 해당 사건번호 순서입니다.

### 다른 버전, total_sumWord 사용


```python
## Word2Vec
import time
startTime = time.time()

word2vec_model = Word2Vec(total_sumWord, size=200,     # 인자값에 new_corpus(948개) 혹은 total_sumWord(63352개)
                          window=2,
                          min_count=10,
                          workers=4,
                          iter=10000, sg=1)

w2v_key = word2vec_model.wv.vocab.keys()
input_keyword_last = ['테스트를 위한 빈값 넣어놓은 것']
print("len(w2v_key):", len(w2v_key))

endTime = time.time() - startTime
print(endTime)
```

    len(w2v_key): 596
    3308.223218202591
    


```python
def input_to_keyword():
    input_text = input("검색어를 입력하세요(끝내시려면 enter 키를 눌러주세요): ")
    k = Komoran()    # 새로 변수를 정의하지 않을시 반복해서 사용자가 검색어를 입력하면 에러가 뜨는 경우가 발생
    if input_text is '':
        input_keyword_last = None
        return input_keyword_last
    else:
        input_corpus = k.pos("\n".join([s for s in input_text.split("\n") if s]))
        parsed_input = append_noun_words(input_corpus)
        input_keyword_last = []
        for j in range(len(parsed_input)):
            # if parsed_input[j] not in word_list:
            # 위에 문장을 쓰지 않는 이유는 사용자가 검색한 키워드가 항상 우리 키워드 안에 있지는 않지만, 사용자의 키워드를 보여줄 필요는 있다.
            input_keyword_last.append(parsed_input[j])
        return input_keyword_last

repeat_num = 1    # 저장할 액셀 파일에 숫자를 붙여주기 위한 기초값, 첫번째를 뜻함.

while True:
    input_keyword_last = input_to_keyword()  # 인풋값을 사용할 키워드로 정리한 값
    if input_keyword_last is None:
        break
    columns_keyword = "·".join(input_keyword_last)  # 컬럼 이름을 위한 합치기

    df = pd.DataFrame()
    predict_words = []
    predict_values = []

    ## TODO: 과연 내가 찾으려고 하는 연관성이 predict_output_word()가 맞는가?
    predict_model = word2vec_model.predict_output_word(input_keyword_last, topn=len(w2v_key))
    for n in range(len(predict_model)):
        predict_words.append((predict_model[n][0]))
        predict_values.append(predict_model[n][1])
    val_words = pd.Series(predict_words)
    val_values = pd.Series(predict_values)
    df[columns_keyword] = val_words
    df['vector값'] = val_values
    #word2vec_model.max_final_vocab  # 이것은 왜 필요한거지? 아마 그냥 들어간거 같은데!!??

    df.to_csv('C:/Users/Jongil Park/PycharmProjects/ai_study/New_projectDirectory_law2/DataSet/Gyotongsago_data_375/Gyotongsago_resultSet'
              '/Gyotongsago_375_W2V_predict({0}).csv'.format(repeat_num),
              encoding='CP949')

    print("({0})차 구동완료".format(repeat_num))

    ## 키워드 연관 단어 상위 20개를 기준으로 한 카운트 값
    df_top20 = df[columns_keyword].iloc[:20]   ## iloc와 loc 차이? 여기에선 같은 값이 나오지만 둘의 차이 명확히 알기
    count_list_total = []
    for kk in range(len(total_panrye_parsingReason)):
        count = 0
        for j in range(len(df_top20)):
            if df_top20[j] in total_panrye_parsingReason[kk]:
                # 현재 count 값은 상위 20개를 순서상관없이 + 값, 순서별로 가중치를 주는 방법도 고려할 것.
                count += total_panrye_parsingReason[kk][df_top20[j]]
        count_list_total.append(count)

    # 카운트 값이 높은 순서의 인덱스 값의 리스트를 만들기
    df_count = pd.DataFrame()
    val_title_list = pd.Series(title_list)
    val_keyNum_list = pd.Series(keyNum_list)

    df_count = pd.DataFrame({(columns_keyword): count_list_total})
    df_count['판례이름'] = val_title_list
    df_count['사건번호'] = val_keyNum_list
    last_df_count = df_count.sort_values(by=columns_keyword, ascending=False)

    last_df_count.to_csv('C:/Users/Jongil Park/PycharmProjects/ai_study/New_projectDirectory_law2/DataSet'
                         '/Gyotongsago_data_375/Gyotongsago_resultSet/Gyotongsago_375_W2V_result({0}).csv'.format(repeat_num),
                         encoding='CP949')
    repeat_num += 1

print("구동완료")
```

    검색어를 입력하세요(끝내시려면 enter 키를 눌러주세요): 도로에서 술을 먹고 음주 운전 뒤 반대편 차량과 부딪쳤습니다
    (1)차 구동완료
    검색어를 입력하세요(끝내시려면 enter 키를 눌러주세요): 
    구동완료
    


```python
df.iloc[:20]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>도로·술·음주·운전·뒤·반대편·차량</th>
      <th>vector값</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>피고인</td>
      <td>0.001678</td>
    </tr>
    <tr>
      <th>1</th>
      <td>반대편</td>
      <td>0.001678</td>
    </tr>
    <tr>
      <th>2</th>
      <td>취득</td>
      <td>0.001678</td>
    </tr>
    <tr>
      <th>3</th>
      <td>유인</td>
      <td>0.001678</td>
    </tr>
    <tr>
      <th>4</th>
      <td>차례</td>
      <td>0.001678</td>
    </tr>
    <tr>
      <th>5</th>
      <td>계약</td>
      <td>0.001678</td>
    </tr>
    <tr>
      <th>6</th>
      <td>앞쪽</td>
      <td>0.001678</td>
    </tr>
    <tr>
      <th>7</th>
      <td>청구</td>
      <td>0.001678</td>
    </tr>
    <tr>
      <th>8</th>
      <td>무시</td>
      <td>0.001678</td>
    </tr>
    <tr>
      <th>9</th>
      <td>공동정범</td>
      <td>0.001678</td>
    </tr>
    <tr>
      <th>10</th>
      <td>면허증</td>
      <td>0.001678</td>
    </tr>
    <tr>
      <th>11</th>
      <td>추락</td>
      <td>0.001678</td>
    </tr>
    <tr>
      <th>12</th>
      <td>손잡이</td>
      <td>0.001678</td>
    </tr>
    <tr>
      <th>13</th>
      <td>특별</td>
      <td>0.001678</td>
    </tr>
    <tr>
      <th>14</th>
      <td>감금</td>
      <td>0.001678</td>
    </tr>
    <tr>
      <th>15</th>
      <td>방어</td>
      <td>0.001678</td>
    </tr>
    <tr>
      <th>16</th>
      <td>대리</td>
      <td>0.001678</td>
    </tr>
    <tr>
      <th>17</th>
      <td>주변</td>
      <td>0.001678</td>
    </tr>
    <tr>
      <th>18</th>
      <td>집행유예</td>
      <td>0.001678</td>
    </tr>
    <tr>
      <th>19</th>
      <td>상호</td>
      <td>0.001678</td>
    </tr>
  </tbody>
</table>
</div>




```python
last_df_count.iloc[:20]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>도로·술·음주·운전·뒤·반대편·차량</th>
      <th>판례이름</th>
      <th>사건번호</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>354</th>
      <td>245</td>
      <td>폭력행위등 처벌에 관한 법률위반(공동공갈)·업무 방해·공갈·폭력행위등 처벌에 관한 ...</td>
      <td>2011노163</td>
    </tr>
    <tr>
      <th>36</th>
      <td>176</td>
      <td>살인(예비적죄명:교통사고처리특례법위반)·사기</td>
      <td>2015노358</td>
    </tr>
    <tr>
      <th>12</th>
      <td>158</td>
      <td>살인(예비적죄명:교통사고처리특례법위반)·사기(남편이 보험금을 노리고 교통사고를 내어...</td>
      <td>2017도1549</td>
    </tr>
    <tr>
      <th>86</th>
      <td>133</td>
      <td>강도치상·특수강도·도로교통법위반(무면허운전)·사기·교통사고처리특례법위반</td>
      <td>2006고합880</td>
    </tr>
    <tr>
      <th>353</th>
      <td>131</td>
      <td>폭력행위등 처벌에 관한 법률위반(공동 공갈)·업무 방해·공갈·폭력행위등 처벌에 관한...</td>
      <td>2009고합613</td>
    </tr>
    <tr>
      <th>373</th>
      <td>129</td>
      <td>감금치사[선택적죄명:살인,인정된죄명:도로교통법위반(사고후미조치)·유기치사]</td>
      <td>2013노2492</td>
    </tr>
    <tr>
      <th>15</th>
      <td>115</td>
      <td>특정범죄가중처벌등에관한법률위반(도주차량)·교통사고처리특례법위반·도로교통법위반·도로교...</td>
      <td>2006노2898</td>
    </tr>
    <tr>
      <th>85</th>
      <td>101</td>
      <td>강도치상(인정된죄명:절도·상해)·특수강도·도로교통법위반(무면허운전)·사기·교통사고처...</td>
      <td>2007노193</td>
    </tr>
    <tr>
      <th>340</th>
      <td>96</td>
      <td>특정범죄가중처벌등에관한법률위반(영리약취·유인등)·아동·청소년의성보호에관한법률위반(강...</td>
      <td>2011노573</td>
    </tr>
    <tr>
      <th>301</th>
      <td>87</td>
      <td>교통사고처리특례법위반</td>
      <td>93노4273</td>
    </tr>
    <tr>
      <th>290</th>
      <td>69</td>
      <td>특정범죄가중처벌등에관한법률위반(도주차량, 인정된 죄명 : 교통사고처리특례법위반)</td>
      <td>96도591</td>
    </tr>
    <tr>
      <th>90</th>
      <td>51</td>
      <td>특정범죄가중처벌등에관한법률위반(도주차량, 인정된 죄명 : 교통사고처리특례법위반)·도...</td>
      <td>99도5023</td>
    </tr>
    <tr>
      <th>287</th>
      <td>48</td>
      <td>교통사고처리특례법위반</td>
      <td>97도1702</td>
    </tr>
    <tr>
      <th>293</th>
      <td>45</td>
      <td>교통사고처리특례법위반</td>
      <td>98도2605</td>
    </tr>
    <tr>
      <th>129</th>
      <td>44</td>
      <td>교통사고처리특례법위반ㆍ업무상과실자동차파괴</td>
      <td>83도3006</td>
    </tr>
    <tr>
      <th>76</th>
      <td>44</td>
      <td>사기·교통사고처리특례법위반·유가증권위조(변경된죄명:유가증권변조)·위조유가증권행사(변...</td>
      <td>2005노396</td>
    </tr>
    <tr>
      <th>281</th>
      <td>44</td>
      <td>살인·폭력행위등처벌에관한법률위반·특수공무집행방해치상·공용물건손상·도로교통법위반·향정...</td>
      <td>96도2588</td>
    </tr>
    <tr>
      <th>57</th>
      <td>42</td>
      <td>교통사고처리특례법위반·도로교통법위반(음주운전)</td>
      <td>2006노1642</td>
    </tr>
    <tr>
      <th>286</th>
      <td>41</td>
      <td>교통사고처리특례법위반·도로교통법위반</td>
      <td>96도1540</td>
    </tr>
    <tr>
      <th>310</th>
      <td>39</td>
      <td>교통사고처리특례법위반</td>
      <td>92도934</td>
    </tr>
  </tbody>
</table>
</div>



### 결과값 중 직관적으로 연관이 높다고 생각했던 결과 예시


```python
# 직관적으로 이해할 수 있는 결과
csv_data_predict = pd.read_csv('C:/Users/Jongil Park/PycharmProjects/ai_study/projectDirectory_law2'
                       '/projectDirectory/POS&embedding/test_dataset/TOTAL_GYOTONGSAGO_DATA/375_GYOTONGSAGO_predict값(2)_2.csv', engine='python') 
csv_data_predict[:20]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>음주·후·음주운전·중·사고·당시·알콜·농도</th>
      <th>vector값</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>알코올</td>
      <td>0.010704</td>
    </tr>
    <tr>
      <th>1</th>
      <td>혈중</td>
      <td>0.009129</td>
    </tr>
    <tr>
      <th>2</th>
      <td>농도</td>
      <td>0.003026</td>
    </tr>
    <tr>
      <th>3</th>
      <td>업무상</td>
      <td>0.001502</td>
    </tr>
    <tr>
      <th>4</th>
      <td>위드마크</td>
      <td>0.001452</td>
    </tr>
    <tr>
      <th>5</th>
      <td>운전사</td>
      <td>0.001273</td>
    </tr>
    <tr>
      <th>6</th>
      <td>교통사고처리</td>
      <td>0.001232</td>
    </tr>
    <tr>
      <th>7</th>
      <td>과실</td>
      <td>0.001214</td>
    </tr>
    <tr>
      <th>8</th>
      <td>공식</td>
      <td>0.001142</td>
    </tr>
    <tr>
      <th>9</th>
      <td>트럭</td>
      <td>0.001118</td>
    </tr>
    <tr>
      <th>10</th>
      <td>특례법</td>
      <td>0.001115</td>
    </tr>
    <tr>
      <th>11</th>
      <td>눈</td>
      <td>0.001082</td>
    </tr>
    <tr>
      <th>12</th>
      <td>태도</td>
      <td>0.001081</td>
    </tr>
    <tr>
      <th>13</th>
      <td>경보</td>
      <td>0.001080</td>
    </tr>
    <tr>
      <th>14</th>
      <td>골목길</td>
      <td>0.001080</td>
    </tr>
    <tr>
      <th>15</th>
      <td>경합</td>
      <td>0.001079</td>
    </tr>
    <tr>
      <th>16</th>
      <td>비상</td>
      <td>0.001078</td>
    </tr>
    <tr>
      <th>17</th>
      <td>일방통행</td>
      <td>0.001078</td>
    </tr>
    <tr>
      <th>18</th>
      <td>강간치상죄</td>
      <td>0.001078</td>
    </tr>
    <tr>
      <th>19</th>
      <td>이탈</td>
      <td>0.001078</td>
    </tr>
  </tbody>
</table>
</div>




```python
csv_data_result = pd.read_csv('C:/Users/Jongil Park/PycharmProjects/ai_study/projectDirectory_law2'
                       '/projectDirectory/POS&embedding/test_dataset/TOTAL_GYOTONGSAGO_DATA/375_GYOTONGSAGO_최종결과(2).csv', engine='python') 
csv_data_result.columns = ['판례번호', '음주·후·음주운전·중·사고·당시·알콜·농도', '판례이름', '사건번호']
csv_data_result[:20]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>판례번호</th>
      <th>음주·후·음주운전·중·사고·당시·알콜·농도</th>
      <th>판례이름</th>
      <th>사건번호</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52</td>
      <td>68</td>
      <td>폭력행위등처벌에관한법률위반(야간·공동상해)·공무집행방해·교통사고처리특례법위반·도로교...</td>
      <td>2004고단232</td>
    </tr>
    <tr>
      <th>1</th>
      <td>95</td>
      <td>65</td>
      <td>교통사고처리특례법위반·도로교통법위반</td>
      <td>2001도1929</td>
    </tr>
    <tr>
      <th>2</th>
      <td>43</td>
      <td>42</td>
      <td>교통사고처리특례법위반·도로교통법위반(음주운전)</td>
      <td>2005도3298</td>
    </tr>
    <tr>
      <th>3</th>
      <td>115</td>
      <td>40</td>
      <td>교통사고처리특례법위반·도로교통법위반(음주운전)</td>
      <td>2008도5531</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37</td>
      <td>35</td>
      <td>교통사고처리특례법위반·도로교통법위반(음주운전)</td>
      <td>2017도661</td>
    </tr>
    <tr>
      <th>5</th>
      <td>57</td>
      <td>33</td>
      <td>교통사고처리특례법위반·도로교통법위반(음주운전)</td>
      <td>2006노1642</td>
    </tr>
    <tr>
      <th>6</th>
      <td>15</td>
      <td>30</td>
      <td>특정범죄가중처벌등에관한법률위반(도주차량)·교통사고처리특례법위반·도로교통법위반·도로교...</td>
      <td>2006노2898</td>
    </tr>
    <tr>
      <th>7</th>
      <td>253</td>
      <td>27</td>
      <td>교통사고처리특례법위반</td>
      <td>91도1746</td>
    </tr>
    <tr>
      <th>8</th>
      <td>262</td>
      <td>25</td>
      <td>교통사고처리특례법위반</td>
      <td>94도2393</td>
    </tr>
    <tr>
      <th>9</th>
      <td>87</td>
      <td>23</td>
      <td>교통사고처리특례법위반</td>
      <td>2007노187</td>
    </tr>
    <tr>
      <th>10</th>
      <td>75</td>
      <td>21</td>
      <td>교통사고처리특례법위반등피고사건</td>
      <td>86노1756</td>
    </tr>
    <tr>
      <th>11</th>
      <td>60</td>
      <td>19</td>
      <td>교통사고처리특례법위반피고사건</td>
      <td>84노95</td>
    </tr>
    <tr>
      <th>12</th>
      <td>168</td>
      <td>18</td>
      <td>교통사고처리특례법위반</td>
      <td>85도1959</td>
    </tr>
    <tr>
      <th>13</th>
      <td>55</td>
      <td>18</td>
      <td>교통사고처리특례법위반</td>
      <td>2006고단1346</td>
    </tr>
    <tr>
      <th>14</th>
      <td>122</td>
      <td>18</td>
      <td>특정범죄가중처벌등에관한법률위반(도주차량)·교통사고처리특례법위반</td>
      <td>83도1328</td>
    </tr>
    <tr>
      <th>15</th>
      <td>28</td>
      <td>18</td>
      <td>교통사고처리특례법위반·도로교통법위반(음주운전)</td>
      <td>2002고단3245</td>
    </tr>
    <tr>
      <th>16</th>
      <td>239</td>
      <td>16</td>
      <td>교통사고처리특례법위반</td>
      <td>90도1873</td>
    </tr>
    <tr>
      <th>17</th>
      <td>36</td>
      <td>16</td>
      <td>살인(예비적죄명:교통사고처리특례법위반)·사기</td>
      <td>2015노358</td>
    </tr>
    <tr>
      <th>18</th>
      <td>197</td>
      <td>15</td>
      <td>교통사고처리특례법위반</td>
      <td>87도249</td>
    </tr>
    <tr>
      <th>19</th>
      <td>167</td>
      <td>15</td>
      <td>교통사고처리특례법위반</td>
      <td>84도2567</td>
    </tr>
  </tbody>
</table>
</div>






![seq2seq](https://user-images.githubusercontent.com/50260643/62867708-ea72c680-bd4e-11e9-8f07-0387e3c2e122.png)



```python
#=============================================================아래부터는 Seq2Seq모델=================================
import sys
import tensorflow as tf
import numpy as np

import cx_Oracle as cx

con=cx.connect('java01/java01@127.0.0.1:1521/xe')# 오라클 connection 획득
cur=con.cursor()
num1='1'#id_num 1번 db 번호
######################################
#           word2vec table save
fornum=0
file="C:/Users/user/PycharmProjects/python_study/Project/375_GYOTONGSAGO_최종결과/375_GYOTONGSAGO_최종결과(1).csv"
infile=open(file)
for line in infile:
    db_seq=line.split(',')
    if db_seq[0] == '':
        cur.execute("insert into word2vec(ID_NUM) values('" + num1 + "')")

        #cur.execute("insert into word2vec(seq) values('" +db_seq[1]+"')")
        cur.execute("update word2vec set seq = '" + db_seq[1] + "' where ID_NUM='" + num1 + "'")
    row1=line.rstrip().split(',')
    if row1[0]!='':
        #print(row1)
        cur.execute("update word2vec set FILENUM = '" + row1[0] + "' where ID_NUM='" + num1 + "'")
        cur.execute("update word2vec set COUNTS = '" + row1[1] + "' where ID_NUM='" + num1 + "'")
        cur.execute("update word2vec set LAW_NAME = '" + row1[2] + "' where ID_NUM='" + num1 + "'")
        cur.execute("update word2vec set LAW_NUMBER = '" + row1[3] + "' where ID_NUM='" + num1 + "'")
        break

    fornum = fornum + 1
    # if fornum==2:
    #     break
    # fornum=fornum+1
    # if fornum==5:
    #     break

##################################################################
#db에 저장된 것을 문자열로 가지고 온다.
#1. db에서 law_number 을 가지고 온다(판례명)
#2. 엑셀 csv 파일의 단어 20개를 가지고 온다.
###################################################################word2vec table > law_number
cur.execute("select law_number from word2vec")
list=cur.fetchall()
Dlaw_number=list[0][0]#<<<<<<<쓸 변수

con.commit()

###############################################################################################
#word2vec에서 나온 벡터단어 리스트를 상위 20개 가져옴
list_375_total=[]
file="C:/Users/user/PycharmProjects/python_study/Project/375_GYOTONGSAGO_최종결과/375_GYOTONGSAGO_predict값(1).csv"
infile3=open(file)
for_number=0
for line in infile3:
    for_number=for_number+1
    if for_number==1:
        continue
    line_list=line.split(',')
    temp=line_list[1].split('\n')
    list_375_total.append(temp)
    if for_number==6:
        break
final_375_word=[] #######################################여기서 필요한 변수 이거 하나 >>>>>>>>>>>>>>>>>>>>>>>>  wordlist 20 개 (상위)
for i in range(5):
    final_375_word.append(list_375_total[i][0])

infile3.close()
#############################################################################
final_375_word=''.join(final_375_word)

x_max=0
y_max=0
seq_data=[]

seq_data.append([Dlaw_number,final_375_word])

# for i in range(len(seq_data)-1):
#     if x_max < len(seq_data[i + 1][0]) and len(seq_data[i][0]) < len(seq_data[i + 1][0]):
#         x_max=len(seq_data[i+1][0])
# for i in range(len(seq_data)-1):
#     if x_max < len(seq_data[i + 1][1]) and len(seq_data[i][0]) < len(seq_data[i + 1][1]):
#         y_max=len(seq_data[i+1][1])

x_max=len(seq_data[0][0])
y_max=len(seq_data[0][1])
###################################################################
#데이터 저장 ( table:seq_lawname_20word)


cur.execute("insert into SEQ_LAWNMBER_20WORD(ID_NUM) values('" + num1 + "')")
cur.execute("update SEQ_LAWNMBER_20WORD set LAW_NUMBER = '" + Dlaw_number + "' where ID_NUM='" + num1 + "'")
cur.execute("update SEQ_LAWNMBER_20WORD set WORDLIST = '" + final_375_word + "' where ID_NUM='" + num1 + "'")

con.commit()
cur.close()
con.close()
#####################################################################################
char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz .가각간갇갈갉갊감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹갼걀걋걍걔걘걜거걱건걷걸걺검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격겪견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괩괬괭괴괵괸괼굄굅굇굉교굔굘굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궜궝궤궷귀귁귄귈귐귑귓규균귤그극근귿글긁금급긋긍긔기긱긴긷길긺김깁깃깅깆깊까깍깎깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄꺅꺌꺼꺽꺾껀껄껌껍껏껐껑께껙껜껨껫껭껴껸껼꼇꼈꼍꼐꼬꼭꼰꼲꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽜꽝꽤꽥꽹꾀꾄꾈꾐꾑꾕꾜꾸꾹꾼꿀꿇꿈꿉꿋꿍꿎꿔꿜꿨꿩꿰꿱꿴꿸뀀뀁뀄뀌뀐뀔뀜뀝뀨끄끅끈끊끌끎끓끔끕끗끙끝끼끽낀낄낌낍낏낑나낙낚난낟날낡낢남납낫났낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냑냔냘냠냥너넉넋넌널넒넓넘넙넛넜넝넣네넥넨넬넴넵넷넸넹녀녁년녈념녑녔녕녘녜녠노녹논놀놂놈놉놋농높놓놔놘놜놨뇌뇐뇔뇜뇝뇟뇨뇩뇬뇰뇹뇻뇽누눅눈눋눌눔눕눗눙눠눴눼뉘뉜뉠뉨뉩뉴뉵뉼늄늅늉느늑는늘늙늚늠늡늣능늦늪늬늰늴니닉닌닐닒님닙닛닝닢다닥닦단닫달닭닮닯닳담답닷닸당닺닻닿대댁댄댈댐댑댓댔댕댜더덕덖던덛덜덞덟덤덥덧덩덫덮데덱덴델뎀뎁뎃뎄뎅뎌뎐뎔뎠뎡뎨뎬도독돈돋돌돎돐돔돕돗동돛돝돠돤돨돼됐되된될됨됩됫됴두둑둔둘둠둡둣둥둬뒀뒈뒝뒤뒨뒬뒵뒷뒹듀듄듈듐듕드득든듣들듦듬듭듯등듸디딕딘딛딜딤딥딧딨딩딪따딱딴딸땀땁땃땄땅땋때땍땐땔땜땝땟땠땡떠떡떤떨떪떫떰떱떳떴떵떻떼떽뗀뗄뗌뗍뗏뗐뗑뗘뗬또똑똔똘똥똬똴뙈뙤뙨뚜뚝뚠뚤뚫뚬뚱뛔뛰뛴뛸뜀뜁뜅뜨뜩뜬뜯뜰뜸뜹뜻띄띈띌띔띕띠띤띨띰띱띳띵라락란랄람랍랏랐랑랒랖랗래랙랜랠램랩랫랬랭랴략랸럇량러럭런럴럼럽럿렀렁렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄롑롓로록론롤롬롭롯롱롸롼뢍뢨뢰뢴뢸룀룁룃룅료룐룔룝룟룡루룩룬룰룸룹룻룽뤄뤘뤠뤼뤽륀륄륌륏륑류륙륜률륨륩륫륭르륵른를름릅릇릉릊릍릎리릭린릴림립릿링마막만많맏말맑맒맘맙맛망맞맡맣매맥맨맬맴맵맷맸맹맺먀먁먈먕머먹먼멀멂멈멉멋멍멎멓메멕멘멜멤멥멧멨멩며멱면멸몃몄명몇몌모목몫몬몰몲몸몹못몽뫄뫈뫘뫙뫼묀묄묍묏묑묘묜묠묩묫무묵묶문묻물묽묾뭄뭅뭇뭉뭍뭏뭐뭔뭘뭡뭣뭬뮈뮌뮐뮤뮨뮬뮴뮷므믄믈믐믓미믹민믿밀밂밈밉밋밌밍및밑바박밖밗반받발밝밞밟밤밥밧방밭배백밴밸뱀뱁뱃뱄뱅뱉뱌뱍뱐뱝버벅번벋벌벎범법벗벙벚베벡벤벧벨벰벱벳벴벵벼벽변별볍볏볐병볕볘볜보복볶본볼봄봅봇봉봐봔봤봬뵀뵈뵉뵌뵐뵘뵙뵤뵨부북분붇불붉붊붐붑붓붕붙붚붜붤붰붸뷔뷕뷘뷜뷩뷰뷴뷸븀븃븅브븍븐블븜븝븟비빅빈빌빎빔빕빗빙빚빛빠빡빤빨빪빰빱빳빴빵빻빼빽뺀뺄뺌뺍뺏뺐뺑뺘뺙뺨뻐뻑뻔뻗뻘뻠뻣뻤뻥뻬뼁뼈뼉뼘뼙뼛뼜뼝뽀뽁뽄뽈뽐뽑뽕뾔뾰뿅뿌뿍뿐뿔뿜뿟뿡쀼쁑쁘쁜쁠쁨쁩삐삑삔삘삠삡삣삥사삭삯산삳살삵삶삼삽삿샀상샅새색샌샐샘샙샛샜생샤샥샨샬샴샵샷샹섀섄섈섐섕서석섞섟선섣설섦섧섬섭섯섰성섶세섹센셀셈셉셋셌셍셔셕션셜셤셥셧셨셩셰셴셸솅소속솎손솔솖솜솝솟송솥솨솩솬솰솽쇄쇈쇌쇔쇗쇘쇠쇤쇨쇰쇱쇳쇼쇽숀숄숌숍숏숑수숙순숟술숨숩숫숭숯숱숲숴쉈쉐쉑쉔쉘쉠쉥쉬쉭쉰쉴쉼쉽쉿슁슈슉슐슘슛슝스슥슨슬슭슴습슷승시식신싣실싫심십싯싱싶싸싹싻싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌜쌤쌥쌨쌩썅써썩썬썰썲썸썹썼썽쎄쎈쎌쏀쏘쏙쏜쏟쏠쏢쏨쏩쏭쏴쏵쏸쐈쐐쐤쐬쐰쐴쐼쐽쑈쑤쑥쑨쑬쑴쑵쑹쒀쒔쒜쒸쒼쓩쓰쓱쓴쓸쓺쓿씀씁씌씐씔씜씨씩씬씰씸씹씻씽아악안앉않알앍앎앓암압앗았앙앝앞애액앤앨앰앱앳앴앵야약얀얄얇얌얍얏양얕얗얘얜얠얩어억언얹얻얼얽얾엄업없엇었엉엊엌엎에엑엔엘엠엡엣엥여역엮연열엶엷염엽엾엿였영옅옆옇예옌옐옘옙옛옜오옥온올옭옮옰옳옴옵옷옹옻와왁완왈왐왑왓왔왕왜왝왠왬왯왱외왹왼욀욈욉욋욍요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅워웍원월웜웝웠웡웨웩웬웰웸웹웽위윅윈윌윔윕윗윙유육윤율윰윱윳융윷으윽은을읊음읍읏응읒읓읔읕읖읗의읜읠읨읫이익인일읽읾잃임입잇있잉잊잎자작잔잖잗잘잚잠잡잣잤장잦재잭잰잴잼잽잿쟀쟁쟈쟉쟌쟎쟐쟘쟝쟤쟨쟬저적전절젊점접젓정젖제젝젠젤젬젭젯젱져젼졀졈졉졌졍졔조족존졸졺좀좁좃종좆좇좋좌좍좔좝좟좡좨좼좽죄죈죌죔죕죗죙죠죡죤죵주죽준줄줅줆줌줍줏중줘줬줴쥐쥑쥔쥘쥠쥡쥣쥬쥰쥴쥼즈즉즌즐즘즙즛증지직진짇질짊짐집짓징짖짙짚짜짝짠짢짤짧짬짭짯짰짱째짹짼쨀쨈쨉쨋쨌쨍쨔쨘쨩쩌쩍쩐쩔쩜쩝쩟쩠쩡쩨쩽쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫓쫘쫙쫠쫬쫴쬈쬐쬔쬘쬠쬡쭁쭈쭉쭌쭐쭘쭙쭝쭤쭸쭹쮜쮸쯔쯤쯧쯩찌찍찐찔찜찝찡찢찧차착찬찮찰참찹찻찼창찾채책챈챌챔챕챗챘챙챠챤챦챨챰챵처척천철첨첩첫첬청체첵첸첼쳄쳅쳇쳉쳐쳔쳤쳬쳰촁초촉촌촐촘촙촛총촤촨촬촹최쵠쵤쵬쵭쵯쵱쵸춈추축춘출춤춥춧충춰췄췌췐취췬췰췸췹췻췽츄츈츌츔츙츠측츤츨츰츱츳층치칙친칟칠칡침칩칫칭카칵칸칼캄캅캇캉캐캑캔캘캠캡캣캤캥캬캭컁커컥컨컫컬컴컵컷컸컹케켁켄켈켐켑켓켕켜켠켤켬켭켯켰켱켸코콕콘콜콤콥콧콩콰콱콴콸쾀쾅쾌쾡쾨쾰쿄쿠쿡쿤쿨쿰쿱쿳쿵쿼퀀퀄퀑퀘퀭퀴퀵퀸퀼큄큅큇큉큐큔큘큠크큭큰클큼큽킁키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탔탕태택탠탤탬탭탯탰탱탸턍터턱턴털턺텀텁텃텄텅테텍텐텔템텝텟텡텨텬텼톄톈토톡톤톨톰톱톳통톺톼퇀퇘퇴퇸툇툉툐투툭툰툴툼툽툿퉁퉈퉜퉤튀튁튄튈튐튑튕튜튠튤튬튱트특튼튿틀틂틈틉틋틔틘틜틤틥티틱틴틸팀팁팃팅파팍팎판팔팖팜팝팟팠팡팥패팩팬팰팸팹팻팼팽퍄퍅퍼퍽펀펄펌펍펏펐펑페펙펜펠펨펩펫펭펴편펼폄폅폈평폐폘폡폣포폭폰폴폼폽폿퐁퐈퐝푀푄표푠푤푭푯푸푹푼푿풀풂품풉풋풍풔풩퓌퓐퓔퓜퓟퓨퓬퓰퓸퓻퓽프픈플픔픕픗피픽핀필핌핍핏핑하학한할핥함합핫항해핵핸핼햄햅햇했행햐향허헉헌헐헒험헙헛헝헤헥헨헬헴헵헷헹혀혁현혈혐협혓혔형혜혠혤혭호혹혼홀홅홈홉홋홍홑화확환활홧황홰홱홴횃횅회획횐횔횝횟횡효횬횰횹횻후훅훈훌훑훔훗훙훠훤훨훰훵훼훽휀휄휑휘휙휜휠휨휩휫휭휴휵휸휼흄흇흉흐흑흔흖흗흘흙흠흡흣흥흩희흰흴흼흽힁히힉힌힐힘힙힛힝1234567890'] #인코딩시켜야한다. 한글일 경우 모든 완성형 문자를 가져와야 한다.

num_dic = {n: i for i, n in enumerate(char_arr)} #딕셔너리 형태로 0,1,2,3, ...  >> 위의 리스트 순서대로 s:0, e:1, p:2, ......짝이 맞춰진다.
dic_len = len(num_dic)
global_step=tf.Variable(0, trainable=False, name='global_step')
#
# seq_data=[]
#
# file="C:/Users/user/PycharmProjects/python_study/Project/test_data.csv"
# infile=open(file)
# for line in infile:
#     seq_data.append(line.split(','))
# for i in range(6):
#     temp=seq_data[i][1][:-1]
#     seq_data[i][1]=temp
# print(seq_data)
#
# print(seq_data[1][0])
# print(seq_data[2][0])
# x_max=0
# y_max=0
# for i in range(len(seq_data)-1):
#     if len(seq_data[i][0]) < len(seq_data[i+1][0]):
#         x_max=len(seq_data[i+1][0])
# for i in range(len(seq_data)-1):
#     if len(seq_data[i][1]) < len(seq_data[i+1][1]):
#         y_max=len(seq_data[i+1][1])
#
# print(x_max)
# print(y_max)
# #
# # #단어 5개 / 교통사고/ 뭐 / 뭐 / 뭐 /뭐 > 판례 가123 > 디비 저장 > 1. 단어를 판례로 검색할 방법을 구해야 한다. 결국 텍스트를 읽어서
# # #상위단어 & 판례 매칭 > 디비저장
# # #러닝 시키고 리스트 추가해서 또 러닝하고 이렇게 할 수 있나??  > 테스트해야 한다. 가능하나 부정확하다.
# # #아니면 데이터들을 계속 모아서 seq2seq돌리고 // 리스트 추가될 때마다 seq2seq 돌리고
# #
# # #  테스트를 통한 결과 (세이브 후 새로운 단어를 누적학습)
# # # 1.학습하고 세이브한 데이터로 재학습이 가능하다.
# # # 2. 다만, 많이 학습한 데이터는 과적합이 발생하고
# # # 3. 새로 들어온 학습데이터는 이미 학습한 단어보다 적은 횟수를 학습하기에 정확도가 떨어진다.
# # #
# # # >>결국, 데이터를 모아서 한번 업데이트 해주는 방법이 좋지 않을까 싶다. 데이터들이 업데이트 될 때마다 하는 것은 문제가 있다.
# #
# #
# #
# #
# # # seq_data = [['조치', '음주운전'], ['당시', '음주운전'], ['여부', '음주운전'],
# # #             ['도망', '도주차량 사고후미조치 음주운전'], ['교통사고', '도주차량 사고후미조치 음주운전'], ['교통', '도주차량 사고후미조치 음주운전']]
# # #########################################################################################

def make_batch(seq_data):
    input_batch = [] #인코더 인풋값
    output_batch = [] #디코더 인풋값
    target_batch = [] #디코더 아웃풋 = y값

    x_max_length=x_max
    y_max_length=y_max+1#심볼 포함 5개 (n+1)

    for seq in seq_data:

        if len(seq) == x_max_length:
            input = [num_dic[n] for n in seq[0]] # 딕셔너리릴 리스트로 바꾼거 // 그리고 seq_data에 한 덩어리씩 가져오면서 그 중 0번째인 word/wood ... 를 가져와서 숫자로 변환한다.
        else:
            input = [num_dic[n] for n in seq[0]]
            for i in range(x_max_length - len(seq[0])):
                input.append(2)

        #input = [num_dic[n] for n in seq[0]]

        if len(seq)==y_max_length:
            output = [num_dic[n] for n in ('S' + seq[1])] #뒤에 y값에 앞에는 심볼인 's'를 붙이고 그 값을 0으로 주면서 나머지는 딕셔너리에 맞는 숫자로 치환한다.
        else:
            output = [num_dic[n] for n in ('S' + seq[1])]
            for i in range(y_max_length - len(seq[1])-1):
                output.append(2)

        if len(seq) == y_max_length:
            target = [num_dic[n] for n in (seq[1] + 'E')]  # 뒤에 y값에 앞에는 심볼인 's'를 붙이고 그 값을 0으로 주면서 나머지는 딕셔너리에 맞는 숫자로 치환한다.
        else:
            target = []
            temp = ([num_dic[n] for n in (seq[1])])
            for i in temp:
                target.append(i)
            target.append(1)

            for i in range(y_max_length - len(seq[1]) - 1):
                target.append(2)

            # target.append([num_dic[n] for n in (seq[1])])

        #target = [num_dic[n] for n in (seq[1] + 'E')] ############디코더에도 한글 넣어주고 출력값도 한글로 표현하는 이유는??? 인코더값을 디코더로 보내는 게 아닌가?

        input_batch.append(np.eye(dic_len)[input]) #np.eye >> n * n 정방행렬을 만들고 값을 주지 않으면 정방위 대각 양수로 위쪽 아래로에 1주려면 음수
        # 41 x 41 행렬의 대각행렬은 가운데를 주고 그 중에서 (input) 인덱스를 가져올 것이다.
        output_batch.append(np.eye(dic_len)[output])
        target_batch.append(target)#sparse_softmax_cross_entropy_with_ligits 사용하면 원핫인코딩 불필요

    return input_batch, output_batch, target_batch

# 요약 : 들어가는 인풋값을 인코더용 디코더용 타겟용으로 잘라놓고
#        원핫 인코딩을 안쓸꺼니까(?) >> 아마도 ep.eye로 배열형태로 0,1로 구분시킨 걸 만들 수 있기때문에
#         np.eye를 사용하고 리스트에 붙여넣는다.

###########################################################################################

learning_rate = 0.0001

n_hidden = 512
total_epoch = 100

n_input = n_class = dic_len #한글 등 배열갯수

enc_input = tf.placeholder(tf.float32, [None, None, n_input])  #shape=[ ,4(none>의미:wood >4),들어가는 갯수가 41개]
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
targets = tf.placeholder(tf.int64, [None, None])

with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)

    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input, #sequence_length=[5],
                                            dtype=tf.float32,)

with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)

    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
                                            initial_state=enc_states, dtype=tf.float32)

model = tf.layers.dense(outputs, n_class, activation=tf.nn.relu, bias_initializer=tf.contrib.layers.xavier_initializer()) #activation func 은 api가 알아서 사용한다. 뭐가 더 좋을지는 돌려보면서 확인해야한다.
#layers를 사용하면 신경망의 값이 들어가고, 노드갯수는 41개다.

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost,global_step=global_step)
saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state('./logs')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    input_batch, output_batch, target_batch = make_batch(seq_data)

    for epoch in range(total_epoch):
        _, loss = sess.run([optimizer, cost],
                           feed_dict={enc_input: input_batch,
                                      dec_input: output_batch,
                                      targets: target_batch})

        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.6f}'.format(loss))
        sys.stdout.flush()

    saver.save(sess, './logs/dnn.ckpt', global_step=global_step)
    print('완료!')

################################################################################################

    def translate(sess, model, word):
        seq_data = [word, 'P' * len(word)]

        input_batch, output_batch, target_batch = make_batch([seq_data])

        prediction = tf.argmax(model, 2)

        result = sess.run(prediction, feed_dict={enc_input: input_batch,
                                                 dec_input: output_batch,
                                                 targets: target_batch})

        decoded = [char_arr[i] for i in result[0]]

        try:
            # first=decoded.insert('P')
            # translated = ''.join(decoded[first:])

            end = decoded.index('E')
            translated = ''.join(decoded[:end])
            #print(translated)
            translated2=""
            for i in translated:
                if i!='P':
                    translated2=translated2+i

            return translated
        except:
            return ''.join(decoded)

    print('\n=== 테스트 ===')
    for i in range(len(seq_data)):
        #print('{0}'.format(seq_data[i][0]), translate(sess, model, seq_data[i][0]))
        print('{0}'.format(seq_data[i][0]), translate(sess, model, seq_data[i][0]))
```


![nextstep](https://user-images.githubusercontent.com/50260643/62868174-ff038e80-bd4f-11e9-866f-065c9ddad8db.png)
