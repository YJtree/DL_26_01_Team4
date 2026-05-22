import pandas as pd
import os

def calculate_top2_distribution(x):
    """
    그룹화된 문장들의 라벨 빈도를 백분율(%)로 계산하여 Top 1, Top 2 라벨과 비율을 반환합니다.
    """
    # normalize=True를 통해 개수가 아닌 비율(0~1)로 계산하고 100을 곱함
    counts = x.value_counts(normalize=True) * 100
    
    # 1순위 라벨 및 비율 추출
    top1_label = counts.index[0] if len(counts) > 0 else "특징 없음"
    top1_ratio = round(counts.iloc[0], 1) if len(counts) > 0 else 0.0
    
    # 2순위 라벨 및 비율 추출 (존재하지 않을 경우 빈칸 처리)
    top2_label = counts.index[1] if len(counts) > 1 else None
    top2_ratio = round(counts.iloc[1], 1) if len(counts) > 1 else 0.0

    return pd.Series({
        'top1_label': top1_label,
        'top1_ratio': top1_ratio,
        'top2_label': top2_label,
        'top2_ratio': top2_ratio
    })

def aggregate_place_labels_advanced(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"[에러] 입력 파일을 찾을 수 없습니다: {input_path}")
        return

    print("1. 문장별 라벨링 데이터 로드 중...")
    df = pd.read_csv(input_path)
    
    print("2. 장소별 분위기 비율(Distribution) 분석 진행 중...")
    # apply 함수를 통해 그룹별로 커스텀 통계 함수(calculate_top2_distribution) 적용
    df_aggregated = df.groupby(['city', 'place_name'])['sentence_label'].apply(calculate_top2_distribution).unstack().reset_index()

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