import pandas as pd
import re
import os

# 1. 텍스트 정제 함수 정의 (줄바꿈 원천 차단 로직 추가됨)
def clean_text(text):
    """
    텍스트 내의 줄바꿈, URL, HTML 태그, 불필요한 특수문자를 모두 제거하고
    한글, 영문, 숫자, 기본 문장 부호만 남기는 핵심 정제 함수입니다.
    """
    if not isinstance(text, str):
        return ""
    
    # 엑셀 셀을 망가뜨리는 엔터키(줄바꿈) 및 캐리지 리턴 기호를 최우선으로 공백 처리
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # 웹 URL 주소 제거
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # 잔여 HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    # 한글, 영문, 숫자, 마침표, 쉼표, 물음표, 느낌표, 공백을 제외한 모든 특수기호 제거
    text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s.,?!]', ' ', text)
    
    # 연속된 다중 공백을 하나의 띄어쓰기로 압축하고 양끝 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 2. 메인 파이프라인 함수
def merge_and_clean_datasets(naver_file, google_file, output_file):
    """
    두 데이터셋의 구조를 통일하여 병합한 뒤, 
    텍스트 정제와 필터링을 수행하고 원본 노이즈 컬럼을 삭제합니다.
    """
    print("[1/5] 데이터 불러오기 시작...")
    df_naver = pd.read_csv(naver_file)
    df_google = pd.read_csv(google_file)
    
    print("[2/5] 데이터 구조 정규화(Normalization) 진행...")
    # 네이버 데이터에 출처 구분을 위한 source 컬럼 추가
    df_naver['source'] = 'naver_blog'
    
    # 구글 데이터에 네이버와 동일한 형태를 맞추기 위해 결측치(None) 컬럼 추가
    df_google['title'] = None
    df_google['link'] = None
    df_google['postdate'] = None
    
    # 컬럼 순서 및 구조 강제 통일
    cols = ['city', 'place_name', 'title', 'link', 'description', 'postdate', 'source']
    df_naver = df_naver[cols]
    df_google = df_google.reindex(columns=cols)
    
    print("[3/5] 데이터 병합(Merge) 진행...")
    # 두 데이터프레임을 상하로 결합
    df_merged = pd.concat([df_naver, df_google], ignore_index=True)
    
    print("[4/5] 텍스트 정제(Cleaning) 및 필터링 진행...")
    # 본문(description)이 비어있는 행 우선 제거
    df_merged = df_merged.dropna(subset=['description'])
    
    # 본문 텍스트에 정제 함수 적용 (결과는 새로운 clean_text 컬럼에 저장)
    df_merged['clean_text'] = df_merged['description'].apply(clean_text)
    
    # 길이 기반 휴리스틱 필터링: 10자 초과 데이터만 보존
    initial_len = len(df_merged)
    df_final = df_merged[df_merged['clean_text'].str.len() > 10]
    filtered_len = len(df_final)
    
    print(f"[5/5] 불필요한 원본 컬럼 삭제 및 최종 데이터 저장 중... (제거된 텍스트: {initial_len - filtered_len}개)")
    
    # 줄바꿈과 노이즈가 섞여 있어 엑셀 뷰를 망가뜨리는 원본 description 컬럼을 완전히 삭제
    df_final = df_final.drop(columns=['description'])
    
    # 결과물을 저장할 폴더 생성 및 CSV 내보내기
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"완료. 총 {filtered_len}개의 정제된 데이터가 '{output_file}'에 저장되었습니다.")

if __name__ == "__main__":
    # 경로 유니코드 에러 방지를 위해 상대 경로를 사용합니다.
    # 현재 실행 위치(YOGIJOGI_JP) 기준의 경로입니다.
    NAVER_PATH = "data/naver_blog_fulltext_all_regions.csv"
    GOOGLE_PATH = "data/google_places_reviews_all_regions.csv"
    OUTPUT_PATH = "data/final_cleaned_merged_data.csv"
    
    merge_and_clean_datasets(NAVER_PATH, GOOGLE_PATH, OUTPUT_PATH)