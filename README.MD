
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


## [Demo 영상]
- 구현 영상의 내용은 1차 모델을 웹으로 구현한 버전
https://www.youtube.com/watch?v=Cz2YCMGPHjQ

## [참고 영상]
- Word2Vec로 벡터화 한 단어를 t-SNE을 통해 시각화
https://www.youtube.com/watch?v=WVAWA-i0v-0




　　
  　
   
   
   
   

![nextstep](https://user-images.githubusercontent.com/50260643/62868174-ff038e80-bd4f-11e9-866f-065c9ddad8db.png)
