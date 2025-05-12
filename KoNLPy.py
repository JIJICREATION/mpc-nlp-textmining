import os
import re
import pandas as pd
from collections import Counter
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. CSV 파일 불러오기 (클렌징된 의사록 텍스트)
csv_path = '/Users/lucia/Desktop/BootCamp/LJKproject/BOK-TEAM-1-LJK-main/lucia/의사록클렌징/2024년 제4차 금통위 의사록_cleaned.csv'
df = pd.read_csv(csv_path)

# 2. KoNLPy의 Okt 분석기 초기화 (stem=True 옵션으로 동사/형용사의 어간을 추출)
okt = Okt()

# 3. 형태소 분석 및 표제어(어간) 추출 함수
def tokenize_and_lemmatize(text):
    # stem=True 옵션 사용으로 동사와 형용사의 어간 추출
    tokens = okt.pos(text, stem=True)
    return tokens

# 4. 불용어 제거 및 특정 품사만 선택 (Okt의 태그: 'Noun', 'Verb', 'Adjective', 'Adverb' 등)
def filter_tokens(tokens, allowed_pos=set(["Noun", "Verb", "Adjective", "Adverb"])):
    # 간단한 불용어 목록 (필요에 따라 확장)
    stopwords = set(["의", "가", "은", "는", "이", "있다", "것", "수", "들"])
    filtered = [word for word, pos in tokens if pos in allowed_pos and word not in stopwords]
    return filtered

# 5. n-gram 생성 (1-gram부터 5-gram)
def generate_ngrams(tokens, n_min=1, n_max=5):
    ngrams = []
    for n in range(n_min, n_max + 1):
        ngrams.extend([' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
    return ngrams

# 6. 문서 전처리: 형태소 분석, 불용어 제거 후 n-gram 생성
def process_text(text):
    tokens = tokenize_and_lemmatize(text)
    filtered_tokens = filter_tokens(tokens)
    ngrams = generate_ngrams(filtered_tokens)
    return filtered_tokens, ngrams

# 7. 각 문서에 대해 토큰과 n-gram 추출
df[['tokens', 'ngrams']] = df['cleaned_text'].apply(lambda x: pd.Series(process_text(x)))

# 8. 전체 코퍼스의 n-gram 빈도 계산
all_ngrams = [ng for ngram_list in df['ngrams'] for ng in ngram_list]
ngram_counts = Counter(all_ngrams)

# 9. 각 문서의 n-gram 중 빈도 15회 미만인 항목 제거
def filter_ngrams_by_freq(ngrams, threshold=15):
    return [ng for ng in ngrams if ngram_counts[ng] >= threshold]

df['ngrams_filtered'] = df['ngrams'].apply(filter_ngrams_by_freq)

# 10. (선택 사항) 겹치는 n-gram 중 가장 긴 것만 남기기
def keep_longest_ngrams(ngrams):
    ngrams_sorted = sorted(ngrams, key=lambda x: len(x.split()), reverse=True)
    selected = []
    for ng in ngrams_sorted:
        if not any(ng in other for other in selected):
            selected.append(ng)
    return selected

df['ngrams_final'] = df['ngrams_filtered'].apply(keep_longest_ngrams)

# 11. 최종 n-gram들을 하나의 문자열로 결합 (각 문서를 하나의 "문서"로 취급)
df['ngram_text'] = df['ngrams_final'].apply(lambda x: ' '.join(x))

# 12. TF-IDF 벡터화: 각 문서의 특성 벡터 생성
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['ngram_text'])
print("TF-IDF 벡터 shape:", X.shape)

# 13. 전처리 결과를 CSV 파일로 저장 (파일 경로와 이름은 필요에 따라 수정)
output_csv = "2024년도_제4차 금통위 의사록_konlpy.csv"
df.to_csv(output_csv, index=False, encoding="utf-8-sig")
