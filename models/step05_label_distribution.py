import pandas as pd
import os

def aggregate_place_labels_advanced(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"[에러] 입력 파일을 찾을 수 없습니다: {input_path}")
        return

    print("1. 문장별 라벨링 데이터 로드 중...")
    df = pd.read_csv(input_path)
    
    print("2. 장소별 모든 분위기 비율(Soft Labeling) 분석 진행 중...")
    
    # 1. groupby를 통해 도시와 장소명으로 그룹화
    # 2. value_counts(normalize=True)로 각 그룹 내 라벨의 상대적 비율(0~1) 계산
    # 3. unstack(fill_value=0)을 통해 라벨 인덱스를 컬럼으로 변환하며, 
    #    해당 장소에 특정 라벨이 아예 없더라도 빈칸(NaN) 대신 0.0을 채워넣어 데이터 유실을 완벽히 방지함
    df_aggregated = (
        df.groupby(['city', 'place_name'])['sentence_label']
        .value_counts(normalize=True)
        .unstack(fill_value=0) * 100
    )
    
    # 소수점 첫째 자리까지 반올림
    df_aggregated = df_aggregated.round(1)
    
    # 컬럼 이름을 직관적으로 변경 (예: 라벨 '3' -> 'label_3_ratio')
    # sentence_label에 어떤 값이 들어있든 동적으로 컬럼명을 생성함
    df_aggregated.columns = [f'label_{col}_ratio' for col in df_aggregated.columns]
    
    # 인덱스를 초기화하여 데이터프레임 구조를 평탄화함
    df_aggregated = df_aggregated.reset_index()

    print("3. 최종 고도화된 추천 데이터베이스 저장 중...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df_aggregated.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("-" * 30)
    print(f"최종 확정된 고유 장소 수: {len(df_aggregated)}개")
    print(f"비율 기반 최종 DB가 '{output_path}'에 저장되었습니다.")

if __name__ == "__main__":
    INPUT_FILE = "data/sentence_labeled_data.csv"
    OUTPUT_FILE = "data/final_travel_database.csv"
    
    aggregate_place_labels_advanced(INPUT_FILE, OUTPUT_FILE)