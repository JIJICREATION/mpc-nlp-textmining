import pandas as pd
import os
import re

# 1. 단일 CSV 파일 경로 지정 (실제 파일 경로로 수정)
input_csv_path = 'test_results_with_predictions.csv'
output_csv_path = 'final_data_with_sentiment.csv'

# 2. CSV 파일 불러오기 (cp949 인코딩 사용)
df = pd.read_csv(input_csv_path, encoding='cp949')

# 3. 불필요한 행 제거 (예시: 'stance' 컬럼에 "매파", "비둘기파", "중립" 단어가 포함된 행 제거)
exclude_words = ['매파', '비둘기파', '중립']
target_column = 'stance'  # 실제 파일에서 해당 컬럼명이 맞는지 확인

if target_column in df.columns:
    mask = df[target_column].astype(str).str.contains('|'.join(exclude_words))
    df = df[~mask]
    print(f"'{target_column}' 컬럼에 {exclude_words} 단어가 포함된 행을 제거했습니다.")
else:
    print(f"'{target_column}' 컬럼이 없으므로, 해당 제거 로직은 적용되지 않습니다.")

# 4. 간단한 긍정/부정/중립 라벨링 (n_gram 컬럼 기반)
n_gram_column = 'n_gram'
if n_gram_column not in df.columns:
    raise ValueError(f"'{n_gram_column}' 컬럼이 존재하지 않습니다. 컬럼명을 확인하세요.")

# 금융 문맥에서 임의로 선정한 긍정/부정 단어 사전
positive_words = ["투자", "개선", "회복", "호전", "상승", "강세", "활성화", "가능성"]
negative_words = ["악화", "축소", "하락", "위축", "불안", "위기", "불확실", "부진"]

def finance_sentiment(ngram):
    """
    n_gram 텍스트에 대해,
    - 부정 단어가 하나라도 있으면 '부정'
    - 긍정 단어가 하나라도 있으면 '긍정'
    - 둘 다 없으면 '중립'
    """
    if pd.isnull(ngram):
        ngram = ""
    if any(word in ngram for word in negative_words):
        return "부정"
    elif any(word in ngram for word in positive_words):
        return "긍정"
    else:
        return "중립"

df['sentiment'] = df[n_gram_column].apply(finance_sentiment)
print("n_gram 컬럼 기반 감성 라벨링을 완료했습니다.")

# 5. 최종 결과 CSV 파일 저장 (cp949 인코딩 사용)
df.to_csv(output_csv_path, index=False, encoding='cp949')
print(f"최종 라벨링 결과가 단일 CSV 파일로 저장되었습니다: {output_csv_path}")
