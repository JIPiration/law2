#-*- coding:utf-8 -*-
## TODO: 만약 모델을 처음부터 다시 만든다면 어떤 가정과 모델을 통해 구현하겠는가? 그리고 그것을 위해 필요한 개념과 공부는 무엇인가?

## TODO: Word2Vec 모델에서 안쓰는 것들 정리한 깔끔한 코드 만들기 용도
""" Think about it!!
TODO: 만약 다시 데이터 전처리를 한다면 어떻게 판례의 어떤 부분을 사용하고('이유'만이 아닌),
TODO: 그리고 각 전체 판례를 합치지 않고 각 판례에서 의미 있는 데이터를 추출하는 것이 옳을까? / 전체 판례를 합쳤을시 올바르지 못한 유사도 값을 가지지 않을까?
TODO: 그 중에서도 어떻게 활용할 것인가(명사만이 아닌) / #혹시 기존 데이터를 다시 학습한다면 각 판례별 이유를 통해 벡터값을 구하는 것이 맞지 않을까??
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
# 왜 이부분에서 오류가 나는 것인지, 이유가 무엇인지 알고 있는가?
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

        # TODO: 특정 처리를 하면 try catch 문장을 쓰지 않아도 될 듯. 텍스트 데이터에 맞는 정규표현식으로 바꾼다면 말이다.
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

## finish 2번 ============================================
## TODO: 3번: 데이터 파일을 통해 Word2Vec 과정
# class Word2Vec:
"""최종적으로 데이터 프레임화를 위한 리스트"""
# 학습데이터 sum 값 - 5개 합치기
## 순서: 교통사고,

# 판례에서 '이유'부분이 의미가 있다고 생각하여 판례를 전처리 해놓은 데이터를 사용
# TODO: 과연 각 키워드(Ex.'교통'...)별 판례의 합으로 전체 단어를 추출하고 그것을 W2V 모델에 넣는 것이 맞을까?
# TODO: 혹은 각 판례를 W2V모델에 넣고 거기서 나온 벡터값이나 키워드 유사성을 합치거나 조절하는 것이 맞을까?
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
    corpus_ko_vocab_items = list(corpus_ko_vocab_items)  ## TODO: 과연 items순이 옳을까?(아니라고 생각한다. 하지만 정말 옳은 방법에 대해 쉽게 감이 잡히지 않는다)
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
        last_using_word.append(''.join(new_corpus[j]))   # join => 리스트를 문자열로 합치기

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
## 직접 Word2Vec 모델을 만들 수 있을까?
print("word2vec 전, 2차 완료")
print("len(total_sumWord: ", len(total_sumWord))
# TODO: Word2Vec의 기본모델의 변수값 설정 Ex. sentence 값, window size, size 등
# TODO: 제목과 가장 연관성 높은 단어 추출 / 2차로 제외시키거나 벡터값을 변형해야 하는 것이 있는지 확인
# TODO: 가장 높은 벡터값의 그룹별 정수 형태로 가장 크게 10 - 0 까지로 나누거나
word2vec_model = Word2Vec(total_sumWord, size=200,     # 인자값에 new_corpus 혹은 word_corpus_last2
                          window=2,
                          min_count=10,
                          workers=4,
                          iter=10000, sg=1)

w2v_key = word2vec_model.wv.vocab.keys()
input_keyword_last = ['테스트를 위한 빈값 넣어놓은 것']
print("len*w2v_key:", len(w2v_key))
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
    word2vec_model.max_final_vocab  # 이것은 왜 필요한거지? 아마 그냥 들어간거 같은데!!??

    df.to_csv("C:/Users/user/PycharmProjects/python_study/projectDirectory/POS&embedding/test_dataset/" \
                "TOTAL_GYOTONGSAGO_DATA/375_GYOTONGSAGO_predict값({0}).csv".format(repeat_num),
              encoding='CP949')

    print("len(predict_model): ", len(predict_model))
    print("({0})차 구동완료".format(repeat_num))

    ## 키워드 연관 단어 상위 20개를 기준으로 한 카운트 값
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

print("구동완료")

## TODO: 여기서 나온 데이터를 활용해서 Seq2Seq 모델에 누적값으로 활용하는 것? // 내가 다시 시퀀스 모델을 만들어 보자
## TODO: 혹은 Seq 모델보다 더 나은 모델은 없는가? 가령 attention 같은

## TODO: 추가적으로 넣을만한 요소나 접근법 등을 다시 고민해볼 것 => 초기 모델 / 수정 모델 결과를 비교할 수 있도록

