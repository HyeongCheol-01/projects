# 웹 문서를 읽어 형태소 분석 : 위키백과에서 단어 검색 결과
# 단어 출력 횟수 DataFrame으로 저장
import requests
from bs4 import BeautifulSoup
from konlpy.tag import Okt
import pandas as pd
from urllib import parse    # 한글 인코딩

okt = Okt()

# url = "https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%88%9C%EC%8B%A0"

para = parse.quote("이순신")
url = "https://ko.wikipedia.org/wiki/" + para
# print(url)

headers = {"User-Agent":"Mozilla/5.0"}

response = requests.get(url, headers=headers)

if response.status_code == 200:     # 요청한 정보 정확히 넘어오면 200
    page = response.text
    # print(page,type(page))  # </html> <class 'str'>
    soup = BeautifulSoup(page, 'lxml')
    
    wordlist = [] # 형태소 분석으로 명사를 추출해 기억
    
    for item in soup.select("#mw-content-text p"):
        if item.string != None:
            wordlist += okt.nouns(item.string)
        
    print('wordlist :', wordlist)
    print('단어 수 :', len(wordlist))
    print("중복 제거 후 단어 수 :", len(set(wordlist)))
    print()
    word_dict = {} # 단어의 발생 횟수를 dict로 저장
    for i in wordlist:
        if i in word_dict:
            word_dict[i] += 1
        else:
            word_dict[i] = 1
            
    print("word_ dict : ", word_dict)

    print('Series로 출력---------')
    seri_list = pd.Series(wordlist)
    print(seri_list[:3])
    print(seri_list.value_counts()[:5])
    print()
    seri_dict = pd.Series(word_dict)
    print(seri_dict[:3])
    print(seri_dict.value_counts()[:5])
    
    print('\nDataFrame으로 출력')
    df1 = pd.DataFrame(wordlist,columns=['단어'])
    print(df1.head(3))
    df2 = pd.DataFrame([word_dict.keys(), word_dict.values()])
    df2 = df2.T
    df2.columns=['단어','빈도수']
    print(df2.head())
    
    df2.to_csv('nlp_morph2.csv', index=False)
    
    df3 = pd.read_csv('nlp_morph2.csv')
    print(df3.head())
    















