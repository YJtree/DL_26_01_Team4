import pandas as pd
import os

# 1. 파일 경로 설정
input_file = "data/naver_blog_crawling_data_large.csv"
output_file = "data/labeling_sample_500.xlsx"

try:
    # 2. 수집해둔 원본 데이터 불러오기
    df = pd.read_csv(input_file)
    print(f"원본 데이터 개수: {len(df)}개")

    # 3. 광고성 키워드 필터링 (결측치 처리 포함)
    # 아래 단어들이 본문(description)에 포함된 글은 삭제합니다.
    ad_keywords = ['소정의 원고료', '제품을 제공받아', '협찬', '업체로부터', '원고료를 지원받아']
    
    # na=False는 내용이 없는 데이터에서 에러가 나는 것을 방지합니다.
    condition = ~df['description'].str.contains('|'.join(ad_keywords), na=False)
    clean_df = df[condition]
    print(f"광고 필터링 후 남은 데이터 개수: {len(clean_df)}개")

    # 4. 수기 라벨링을 위한 무작위 샘플링 (500개 추출)
    # 남은 데이터가 500개보다 적다면 있는 만큼만 추출합니다.
    sample_size = min(500, len(clean_df))
    # random_state=42를 주면 코드를 여러 번 실행해도 항상 똑같은 500개가 뽑힙니다.
    sampled_df = clean_df.sample(n=sample_size, random_state=42) 

    # 5. 조원들이 직접 입력할 '정답지' 빈 칸(칼럼) 만들기
    sampled_df['분위기_라벨(직접입력)'] = ""

    # 6. 팀원 공유용 엑셀 파일(.xlsx)로 저장
    sampled_df.to_excel(output_file, index=False)
    print(f"라벨링용 데이터 추출 성공! '{output_file}' 파일이 생성되었습니다.")

except FileNotFoundError:
    print(f"에러: '{input_file}' 파일을 찾을 수 없습니다. 크롤링부터 다시 확인해 주세요.")