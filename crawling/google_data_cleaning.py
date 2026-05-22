import pandas as pd
import re
import os

def clean_google_reviews():
    input_file = "data/google_places_reviews.csv"
    output_file = "data/google_places_reviews_clean.csv"

    if not os.path.exists(input_file):
        print(f"'{input_file}' 파일이 없습니다. 수집 코드를 먼저 실행해주세요.")
        return

    # 1. 데이터 불러오기
    df = pd.read_csv(input_file)
    initial_count = len(df)
    print(f"초기 수집된 데이터 개수: {initial_count}개")

    # 2. 결측치 및 완전히 똑같은 중복 리뷰 제거
    df = df.dropna(subset=['description'])
    df = df.drop_duplicates(subset=['description'])

    # 3. 너무 짧은 리뷰 제거 (최소 10자 이상만 남김)
    # "좋아요", "맛있어요" 같은 단답형으로는 감성을 유추할 수 없기 때문입니다.
    df = df[df['description'].str.len() >= 10]

    # 4. 한국어(한글)가 포함되지 않은 외국어 리뷰 완벽 차단
    # 정규표현식을 사용하여 한글[가-힣]이 한 글자라도 있는지 검사합니다.
    def has_korean(text):
        return bool(re.search(r'[가-힣]', str(text)))
    
    df = df[df['description'].apply(has_korean)]

    # 5. 보기 싫은 줄바꿈(\n)을 공백으로 교체하여 한 줄로 깔끔하게 만들기
    df['description'] = df['description'].str.replace(r'\n+', ' ', regex=True)

    # 6. 결과 저장
    clean_count = len(df)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"정제 완료! 외국어/단답형 등 {initial_count - clean_count}개의 쓰레기 데이터를 버렸습니다.")
    print(f"살아남은 {clean_count}개의 깨끗한 데이터가 '{output_file}'에 저장되었습니다.")

if __name__ == "__main__":
    clean_google_reviews()