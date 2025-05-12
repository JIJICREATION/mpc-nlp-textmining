import os
import re
import pdfplumber
import pandas as pd

# 클렌징할 특정 단어 목록
words_to_remove = [""]

# PDF 파일 경로 설정
pdf_dir = pdf_dir = "C:/Users/abwm2/Desktop/BootCamp/TIL/JIMINSEONG/jiminseong/BOK_Minutes2"


# PDF 파일 처리
for pdf_file in os.listdir(pdf_dir):
    if pdf_file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_dir, pdf_file)

        # pdfplumber를 사용하여 PDF 파일 열기
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text() + '\n'  # 각 페이지의 텍스트 추출

            # 클렌징: 숫자, 한글, 영어만 남기기
            text_cleaned = re.sub(r'[^0-9가-힣a-zA-Z\s]', '', text)

            # 특정 단어 제거
            for word in words_to_remove:
                text_cleaned = text_cleaned.replace(word, '')

            # 공백 정리
            text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()

            # 클렌징된 텍스트를 데이터프레임으로 변환
            df = pd.DataFrame({'cleaned_text': [text_cleaned]})

            # 각 PDF 파일에 대한 CSV 파일 저장
            output_csv = os.path.join(pdf_dir, f"{os.path.splitext(pdf_file)[0]}_cleaned.csv")
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")

        print(f"{pdf_file} 클렌징 완료. 결과는 '{output_csv}'에 저장되었습니다.")

