import pandas as pd
import os

def extract_remaining_data():
    # 1. 파일 경로 설정 (실제 파일명과 확장자에 맞게 수정하세요)
    # 원본 데이터는 CSV 형식이므로 확장자를 .csv로 변경합니다.
    raw_data_path = "data/naver_blog_crawling_data_large.csv" 
    
    # 조원들이 라벨링한 파일은 엑셀 형식을 유지합니다.
    labeled_data_path = "data/final_integrated_labeling.xlsx" 
    
    if not os.path.exists(raw_data_path) or not os.path.exists(labeled_data_path):
        print("원본 파일 또는 라벨링 완료 파일을 찾을 수 없습니다. 경로와 확장자를 확인해주세요.")
        return

    print("데이터를 불러오는 중입니다...")
    
    # [수정됨] 원본 데이터는 read_csv로 읽습니다. 
    # encoding='utf-8-sig'는 한글 데이터가 깨지지 않도록 방지하는 옵션입니다.
    # 만약 에러가 난다면 encoding='cp949' 로 변경해 보세요.
    df_raw = pd.read_csv(raw_data_path, encoding='utf-8-sig')
    
    # 라벨링 데이터는 기존처럼 read_excel로 읽습니다.
    df_labeled = pd.read_excel(labeled_data_path)

    # 2. 중복 제거 (차집합 연산)
    # 리뷰 텍스트가 완전히 동일한 것을 찾아 원본에서 제외합니다.
    df_remaining = df_raw[~df_raw['review_text'].isin(df_labeled['review_text'])]

    print(f"전체 원본 데이터: {len(df_raw)}개")
    print(f"이미 라벨링한 데이터: {len(df_labeled)}개")
    print(f"새로 탐색할 수 있는 남은 데이터: {len(df_remaining)}개")

    # 3. 새로운 엑셀 파일로 저장
    # 조원들이 작업하기 편하도록 추출된 결과물은 엑셀 파일로 만들어 줍니다.
    output_path = "data/remaining_to_label.xlsx"
    df_remaining.to_excel(output_path, index=False)
    print(f"\n성공! 남은 데이터가 '{output_path}'에 저장되었습니다.")

if __name__ == "__main__":
    extract_remaining_data()