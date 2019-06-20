#-*- coding:utf-8 -*-

""" Think about it!!
TODO: 만약 다시 데이터 전처리를 한다면 어떻게 판례의 어떤 부분을 사용하고('이유'만이 아닌),
TODO: 그리고 각 전체 판례를 합치지 않고 각 판례에서 의미 있는 데이터를 추출하는 것이 옳을까? / 전체 판례를 합쳤을시 올바르지 못한 유사도 값을 가지지 않을까?
TODO: 그 중에서도 어떻게 활용할 것인가(명사만이 아닌)
"""
## TODO: 토탈'교통사고'관련 전체 판례(약 2000개)를 통한 학습 토탈 코드
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import numpy as np
from pprint import pprint
from konlpy.tag import Komoran;
k = Komoran()   #TODO: Q.코모란 사용 이유?
import nltk   #TODO: Q.nltk란 무엇인가?
from nltk import FreqDist  # FreqDist 클래스는 문서에 사용된 단어(토큰)의 사용빈도 정보를 담는 클래스
import pandas as pd
import codecs


## '토탈교통사고 일 때 Set
# 1차적으로 사용하지 않을 키워드, 판례에서 필요한 부분(판례 제목, 사건 번호 등을 따로 뽑아 놓음)
fileName_dnusing_wordSet = 'C:/Users/user/PycharmProjects/python_study/projectDirectory/POS&embedding/' \
                 'test_dataset/TOTAL_GYOTONGSAGO_DATA/TOTAL_GYOTONGSAGO_dnUsingWordSet_2000_onlyNoun.txt'
fileName_title = 'C:/Users/user/PycharmProjects/python_study/projectDirectory/POS&embedding/' \
                 'test_dataset/TOTAL_GYOTONGSAGO_DATA/TOTAL_GYOTONGSAGO_lawTitle.txt'
fileName_keyNum = 'C:/Users/user/PycharmProjects/python_study/projectDirectory/POS&embedding/' \
                 'test_dataset/TOTAL_GYOTONGSAGO_DATA/TOTAL_GYOTONGSAGO_lawNumber.txt'

## 각 판례에서 빼야 할 단어를 위한 리스트: word_list
## TODO: Q.word_list를 나눈 기준은 무엇인가? 그것을 사람의 기준으로 나누고 시작하는 것이 과연 올바른 word2vec을 위한 전처리인가?
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

## 판례의 명사 키워드의 단어:갯수 매칭된 리스트
total_panrye_pasingReason = []  # 이 안의 값은 딕셔너리가 되어야 함

## 여기서 pasing 된 corpus 값을 명사만 추출하여 각 단어별 갯수를 딕셔너리로 만들어주고 append
## TODO: 과연 이 데이터에서 명사만을 추출하는 것이 올바른 접근법인가? 이것으로 명사들과의 관계나 유사도를 올바르게 판단할 수 있는가?
def append_noun_words(corpus):
    noun_words = ['NNG', 'NNB', 'NP']  # 일반명사, 고유명사, 대명사만 학습
    results = []
    for text in corpus:
        for noun_word in noun_words:
            if noun_word in text[1]:
                results.append(text[0])
    return results

file_name = ["C:/Users/user/PycharmProjects/python_study/projectDirectory/POS&embedding/test_dataset/"
            "TOTAL_GYOTONGSAGO_DATA/TOTAL_GYOTONGSAGO_eachReason_2000/교통사고/({0}).txt",
            "C:/Users/user/PycharmProjects/python_study/projectDirectory/POS&embedding/test_dataset/"
            "TOTAL_GYOTONGSAGO_DATA/TOTAL_GYOTONGSAGO_eachReason_2000/교통/({0}).txt",
            "C:/Users/user/PycharmProjects/python_study/projectDirectory/POS&embedding/test_dataset/"
            "TOTAL_GYOTONGSAGO_DATA/TOTAL_GYOTONGSAGO_eachReason_2000/사고/({0}).txt",
            "C:/Users/user/PycharmProjects/python_study/projectDirectory/POS&embedding/test_dataset/"
            "TOTAL_GYOTONGSAGO_DATA/TOTAL_GYOTONGSAGO_eachReason_2000/운전/({0}).txt",
            "C:/Users/user/PycharmProjects/python_study/projectDirectory/POS&embedding/test_dataset/"
            "TOTAL_GYOTONGSAGO_DATA/TOTAL_GYOTONGSAGO_eachReason_2000/차량/({0}).txt"]
file_len = [375, 956, 382, 329, 229]  ## TODO: 과연 텍스트 파일에서 전체 길이를 파악할 수 있도록 하려면 어떻게 해야할까?

for z in range(len(file_name)):

    for i in range(file_len[z]):    # len(title_list)
        ione_panrye_pasingReason = {}

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
                ione_panrye_pasingReason[corpus_vocab_common[j][0]] = corpus_vocab_common[j][1]
        total_panrye_pasingReason.append(ione_panrye_pasingReason)

print("1차 완료: 각 타이틀, 사건번호, 딕셔너리(단어: 번호)매칭")
"""
이것은 전체 corpus로 학습된 데이터 테스트를 위한 word_list 단어 제외시키지 않은 모든 명사들의 합 가지고 있기
    for j in range(len(corpus_vocab_common)):
        ione_panrye_pasingReason[corpus_vocab_common[j][0]] = corpus_vocab_common[j][1]
    total_panrye_pasingReason.append(ione_panrye_pasingReason)
"""

## finish 2번 ============================================
## TODO: 3번: 데이터 파일을 통해 Word2Vec 과정
# class Word2Vec:
"""최종적으로 데이터 프레임화를 위한 리스트"""
# 학습데이터 sum 값 - 5개 합치기
## 순서: 교통사고,
"""
## 중복되는 단어도 살리는 단어 전처리
data_file = ["C:/Users/user/PycharmProjects/python_study/projectDirectory/POS&embedding/test_dataset/" \
            "TOTAL_GYOTONGSAGO_DATA/TOTAL_GYOTONGSAGO_sumReason_5/NONAMEYET.txt"]
total_sumWord = []
for i in range(len(data_file)):
    try:
        with open(data_file[i]) as f:
            texts = f.read()
            corpus = k.pos("\n".join([s for s in texts.split("\n") if s]))

    except UnicodeDecodeError:
        with codecs.open(data_file[i], "r", "utf-8") as f:
            texts = f.read()
            corpus = k.pos("\n".join([s for s in texts.split("\n") if s]))

    corpus = append_noun_words(corpus)
    corpus_ko = nltk.Text(corpus, name="각 판례별 명사 처리")
    corpus_ko_vocab = corpus_ko.vocab()
    corpus_vocab_common = corpus_ko_vocab.most_common(len(corpus_ko))

    ## 쓰는 단어와 그 갯수를 확인하기 위한
    new_corpus = []
    corpus_values = []
    new_corpus2 = []

    for z in range(len(corpus_vocab_common)):
        corpus_word = corpus_vocab_common[z][0]
        if corpus_word not in word_list:
            new_corpus.append(corpus_word.split())
            corpus_values.append(corpus_vocab_common[z][1])
    new_corpus2 = sum(new_corpus, [])

    last_using_word = []
    for j in range(len(new_corpus)):
        last_using_word.append(''.join(new_corpus[j]))

    df3 = pd.DataFrame(corpus_values, index=last_using_word)
    df3.to_excel("C:/Users/user/PycharmProjects/python_study/projectDirectory/POS&embedding/test_dataset/" \
                 "TOTAL_GYOTONGSAGO_DATA/TOTAL_GYOTONGSAGO_USING_WORD_LAST.xlsx",
                 sheet_name='{0}번째'.format(i))

    ## 이것이 corpus 에서 쓰지 않은 키워드를 갯수만큼 제외시키는 것
    word_corpus_last = [x for x in corpus if x not in word_list]
    word_corpus_last2 = []
    for z in range(len(word_corpus_last)):
        word_corpus_last2.append(word_corpus_last[z].split())

    total_sumWord.append(word_corpus_last2)
total_sumWord = sum(total_sumWord, [])
"""
# 판례에서 '이유'부분이 의미가 있다고 생각하여 판례를 전처리 해놓은 데이터를 사용
data_file = ["C:/Users/user/PycharmProjects/python_study/projectDirectory/POS&embedding/test_dataset/"
            "TOTAL_GYOTONGSAGO_DATA/TOTAL_GYOTONGSAGO_sumReason_5/(1).txt",
            "C:/Users/user/PycharmProjects/python_study/projectDirectory/POS&embedding/test_dataset/"
            "TOTAL_GYOTONGSAGO_DATA/TOTAL_GYOTONGSAGO_sumReason_5/(2).txt",
            "C:/Users/user/PycharmProjects/python_study/projectDirectory/POS&embedding/test_dataset/"
            "TOTAL_GYOTONGSAGO_DATA/TOTAL_GYOTONGSAGO_sumReason_5/(3).txt",
            "C:/Users/user/PycharmProjects/python_study/projectDirectory/POS&embedding/test_dataset/"
            "TOTAL_GYOTONGSAGO_DATA/TOTAL_GYOTONGSAGO_sumReason_5/(4).txt",
            "C:/Users/user/PycharmProjects/python_study/projectDirectory/POS&embedding/test_dataset/"
            "TOTAL_GYOTONGSAGO_DATA/TOTAL_GYOTONGSAGO_sumReason_5/(5).txt"]

# 중복되는 단어를 제외한 하나의 키워드만을 사용하는 방식
## 하지만 순서는 고려할 수 있도록
total_sumWord = []
for i in range(len(data_file)):
    try:
        with open(data_file[i]) as f:
            texts = f.read()
            corpus = k.pos("\n".join([s for s in texts.split("\n") if s]))

    except UnicodeDecodeError:
        with codecs.open(data_file[i], "r", "utf-8") as f:
            texts = f.read()
            corpus = k.pos("\n".join([s for s in texts.split("\n") if s]))

    corpus = append_noun_words(corpus)
    corpus_ko = nltk.Text(corpus, name="각 판례별 명사 처리")
    corpus_ko_vocab = corpus_ko.vocab()
    corpus_ko_vocab_items = corpus_ko_vocab.items()
    corpus_ko_vocab_items = list(corpus_ko_vocab_items)   ## TODO: 과연 items순이 옳을까?(아니라고 생각한다. 하지만 정말 옳은 방법에 대해 쉽게 감이 잡히지 않는다)
    #corpus_vocab_common = corpus_ko_vocab.most_common(len(corpus_ko))

    ## TODO: 전체 단어의 순서를 유지하지 않는다면 굳이 전체 데이터를 사용할 필요나 이유가 없지 않은가?
    ## TODO: 그렇다면 전체 순서를 유지하면서 어떻게 우리가 원하는 올바른 유사도를 가진 데이터로 만들 수 있을까?

    ## new_corpus 는 살릴 단어가 하나씩 밖에 안 들어가 있다. 그 개수가 빠져있는 합.
    new_corpus = []
    corpus_values = []
    new_corpus2 = []

    for z in range(len(corpus_ko_vocab_items)):
        corpus_word = corpus_ko_vocab_items[z][0]
        if corpus_word not in word_list:
            new_corpus.append([corpus_word])
            corpus_values.append(corpus_ko_vocab_items[z][1])
    new_corpus2 = sum(new_corpus, [])

    last_using_word = []
    for j in range(len(new_corpus)):
        last_using_word.append(''.join(new_corpus[j]))

    df3 = pd.DataFrame(corpus_values, index=last_using_word)
    df3.to_excel("C:/Users/user/PycharmProjects/python_study/projectDirectory/POS&embedding/test_dataset/" \
                 "TOTAL_GYOTONGSAGO_DATA/TOTAL_GYOTONGSAGO_USING_WORD_LAST({0}).xlsx".format(i+1),
                 sheet_name='단어')
    print("TOTAL_GYOTONGSAGO_USING_WORD_LAST 저장완료")

    ## 이것이 corpus 에서 쓰지 않은 키워드를 갯수만큼 제외시키는 것
    word_corpus_last = [x for x in corpus if x not in word_list]
    word_corpus_last2 = []
    for z in range(len(word_corpus_last)):
        word_corpus_last2.append([word_corpus_last[z]])   # word_corpus_last2.append(word_corpus_last[z].split())에서 변경

    total_sumWord.append(word_corpus_last2)
    # total_sumWord.append(new_corpus)

total_sumWord = sum(total_sumWord, [])
df4 = pd.DataFrame(total_sumWord)
df4.to_excel("C:/Users/user/PycharmProjects/python_study/projectDirectory/POS&embedding/test_dataset/" \
             "TOTAL_GYOTONGSAGO_DATA/TOTAL_GYOTONGSAGO_USING_WORD_LAST_하나로.xlsx",
             sheet_name='단어')
## Word2Vec
print("word2vec 전, 2차 완료")
# TODO: Word2Vec의 기본모델의 변수값 설정 Ex. sentence 값, window size, size 등
# TODO: 제목과 가장 연관성 높은 단어 추출 / 2차로 제외시키거나 벡터값을 변형해야 하는 것이 있는지 확인
# TODO: 가장 높은 벡터값의 그룹별 정수 형태로 가장 크게 10 - 0 까지로 나누거나
word2vec_model = Word2Vec(total_sumWord, size=200,     # 인자값에 new_corpus 혹은 word_corpus_last2
                          window=2,
                          min_count=10,
                          workers=4,
                          iter=10000, sg=1)

w2v_key = word2vec_model.wv.vocab.keys()
### 자동화: titles로 맨윗줄만 추출해서 네이밍 리스트로 바꿀 것

input_keyword_last = ['테스트를 위한 빈값 넣어놓은 것']

def input_to_keyword():
    input_text = input("검색어를 입력하세요(끝내시려면 enter 키를 눌러주세요): ")
    k = Komoran()    # 새로 변수를 정의하지 않을시 반복해서 입력하면 에러가 뜨는 경우가 발생
    if input_text is '':
        input_keyword_last = None
        return input_keyword_last
    else:
        input_corpus = k.pos("\n".join([s for s in input_text.split("\n") if s]))
        parsed_input = append_noun_words(input_corpus)
        input_keyword_last = []
        for j in range(len(parsed_input)):
            # if parsed_input[j] not in word_list:
            input_keyword_last.append(parsed_input[j])
        return input_keyword_last

repeat_num = 1

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
    word2vec_model.max_final_vocab  # 이것은 왜 필요한거지? 아마 그냥 들어간거 같은데!!??

    df.to_csv("C:/Users/user/PycharmProjects/python_study/projectDirectory/POS&embedding/test_dataset/" \
                "TOTAL_GYOTONGSAGO_DATA/375_GYOTONGSAGO_predict값({0}).csv".format(repeat_num),
              encoding='CP949')

    print("len(predict_model): ", len(predict_model))
    print("({0})차 구동완료".format(repeat_num))

    ## 키워드 연관 단어 상위 10개를 기준으로 한 카운트 값
    df_top10 = df[columns_keyword].iloc[:20]   ## iloc와 loc 차이? 여기에선 같은 값이 나오지만 둘의 차이 명확히 알기
    count_list_total = []
    for kk in range(len(total_panrye_pasingReason)):
        count = 0
        for j in range(len(df_top10)):
            if df_top10[j] in total_panrye_pasingReason[kk]:
                count += total_panrye_pasingReason[kk][df_top10[j]]
        count_list_total.append(count)

    # 카운트 값이 높은 순서의 인덱스 값의 리스트를 만들기
    df_count = pd.DataFrame()
    val_title_list = pd.Series(title_list)
    val_keyNum_list = pd.Series(keyNum_list)

    df_count = pd.DataFrame({(columns_keyword): count_list_total})
    df_count['판례이름'] = val_title_list
    df_count['사건번호'] = val_keyNum_list
    last_df_count = df_count.sort_values(by=columns_keyword, ascending=False)

    last_df_count.to_csv("C:/Users/user/PycharmProjects/python_study/projectDirectory/POS&embedding/test_dataset/" \
                           "TOTAL_GYOTONGSAGO_DATA/375_GYOTONGSAGO_최종결과({0}).csv".format(repeat_num),
                         encoding='CP949')
    repeat_num += 1


# 여기서 나온 최종 결고를 통해 PCA로 구현하는 코드 저장을 하지 않았다.
# 각 글자에 따른 벡터값을 메모장이나 액셀에 정리해서 다시 워크패드를 통해 차원을 정리했었나?
# 오쨌든 각 단어와 그에 따른 벡터값의 데이터만 있으면 PCA와 프로젝터로 보여줄 수 있는 구성은 만들 수 있다
#  다만 저장을 하지 않아놔서 다시 또 코드를 만들어야 하는 것이 비효율적인 것 뿐이다.
# 하긴 근데 애초에 위의 데이터가 내가 원하는 연관성이 적은 데이터라면 애초에 굳이 그것을 시각화할 필요가 없을수도 있다

print("구동완료")


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