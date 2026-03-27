# 한글 형태소 분석
# 코퍼스(Corpus, 말뭉치)는 언어 연구, AI 학습, 자연어 처리(NLP)를 목적으로 
# 실제 사용된 언어(글, 말)를 컴퓨터가 읽을 수 있는 형태로 대규모로 수집, 가공, 저장한 
# 언어 자료의 집합.
# 문법 연구, 번역 시스템, 챗봇 등 다양한 언어 데이터 분석의 기초 자료로 활용
# 형태소(Morpheme): 의미를 가지는 가장 작은 단위

# 대표적인 한글 형태소 분석 라이브러리
from konlpy.tag import Okt, Kkma, Komoran

text = "나는 오늘 아침에 학교에 갔다. 가는 길에 벚꽃이 피어 너무 아름다웠다."

print("Okt ----------")
okt=Okt()
print("형태소 : ", okt.morphs(text))
print("품사 태깅 : ", okt.pos(text))
print("품사 태깅(어간 포함) :", okt.pos(text, stem=True))
print("명사 추출 : ", okt.nouns(text))

print("Kkma ----------------")
kkma = Kkma()
print("형태소 : ", kkma.morphs(text))
print("품사 태깅 : ", kkma.pos(text))
print("명사 추출 : ", kkma.nouns(text))


print("Komoran ----------------")
komoran=Komoran()
print("형태소 : ", komoran.morphs(text))
print("품사 태깅 : ", komoran.pos(text))
print("명사 추출 : ", komoran.nouns(text))












