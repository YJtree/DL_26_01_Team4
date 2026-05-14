import pandas as pd
import os

def merge_cleaned_datasets():
    # 파일 경로 설정
    naver_path = "data/labeling_sample_500.xlsx"
    google_path = "data/google_places_reviews_clean.csv"
    output_path = "data/final_integrated_labeling.xlsx"

    # 파일 존재 여부 확인
    if not os.path.exists(naver_path) or not os.path.exists(google_path):
        print("파일을 찾을 수 없습니다. 경로와 파일명을 확인해주세요.")
        return

    # 1. 데이터 불러오기 (네이버는 엑셀, 구글은 CSV)
    df_naver = pd.read_excel(naver_path)
    df_google = pd.read_csv(google_path)

    # 2. 기둥 이름 통일 (Mapping)
    # 네이버와 구글의 기존 기둥 이름을 확인하여 place_name과 review_text로 변경합니다.
    # 각 파일의 실제 기둥 이름에 맞춰 아래 rename 딕셔너리를 수정하세요.
    df_naver = df_naver.rename(columns={
        'title': 'place_name', 
        'description': 'review_text'
    })
    
    df_google = df_google.rename(columns={
        'keyword': 'place_name', 
        'description': 'review_text'
    })

    # 3. 필요한 기둥만 선택하여 합치기
    # 출처(source) 기둥이 있다면 유지하고, 없다면 생성합니다.
    if 'source' not in df_naver.columns:
        df_naver['source'] = 'Naver'
    if 'source' not in df_google.columns:
        df_google['source'] = 'Google'

    selected_columns = ['source', 'place_name', 'review_text']
    
    # 두 데이터프레임을 위아래로 연결
    df_combined = pd.concat([
        df_naver[selected_columns], 
        df_google[selected_columns]
    ], ignore_index=True)

    # 4. 중복 제거
    # 혹시 모를 중복된 리뷰 내용을 제거합니다.
    df_combined = df_combined.drop_duplicates(subset=['review_text'])

    # 5. 라벨링용 빈 기둥 추가
    # 조원들이 숫자를 입력할 label 기둥을 만듭니다.
    df_combined['label'] = ""

    # 6. 최종 엑셀 저장
    df_combined.to_excel(output_path, index=False)
    print(f"통합 완료. 총 {len(df_combined)}건의 데이터가 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    merge_cleaned_datasets()