import pandas as pd
import os

def aggregate_place_labels_advanced(input_path, output_path, min_sentence_count=5):
    if not os.path.exists(input_path):
        print(f"[에러] 입력 파일을 찾을 수 없습니다: {input_path}")
        return

    print("1. 문장별 라벨링 데이터 로드 중...")
    df = pd.read_csv(input_path)
    
    # =========================================================================
    # 최빈값의 통계적 불안정성 제거
    # 해당 장소의 유효 리뷰 문장이 최소 N개(기본값 5개) 이상인 곳만 필터링합니다.
    # =========================================================================
    print(f"2. 통계적 신뢰도 확보: 리뷰 문장 임계값({min_sentence_count}개 이상) 필터링 중...")
    
    # 장소별 전체 문장 수 계산
    place_counts = df.groupby(['city', 'place_name']).size().reset_index(name='sentence_count')
    
    # 지정된 임계값(Threshold)을 넘는 장소만 필터링 (신뢰할 수 있는 장소)
    valid_places = place_counts[place_counts['sentence_count'] >= min_sentence_count]
    
    # 원본 데이터프레임과 병합하여 유효한 장소의 문장들만 남김
    filtered_df = df.merge(valid_places[['city', 'place_name']], on=['city', 'place_name'], how='inner')
    
    # 터미널에 필터링 결과 출력 (발표 및 보고서 작성 시 정량적 근거로 활용 가능)
    dropped_count = len(place_counts) - len(valid_places)
    print(f"   - 초기 전체 장소 수: {len(place_counts)}개")
    print(f"   - 제외된 장소 수 (리뷰 {min_sentence_count}개 미만): {dropped_count}개")
    print(f"   - 최종 분석 대상 (신뢰 장소): {len(valid_places)}개")

    print("3. 장소별 모든 분위기 비율(Soft Labeling) 분석 진행 중...")
    # 1. groupby를 통해 도시와 장소명으로 그룹화
    # 2. value_counts(normalize=True)로 각 그룹 내 라벨의 상대적 비율(0~1) 계산
    # 3. unstack(fill_value=0)을 통해 특정 라벨이 아예 없더라도 0.0으로 빈칸을 채움
    df_aggregated = (
        filtered_df.groupby(['city', 'place_name'])['sentence_label']
        .value_counts(normalize=True)
        .unstack(fill_value=0) * 100
    )
    
    # 소수점 첫째 자리까지 반올림
    df_aggregated = df_aggregated.round(1)
    
    # 컬럼 이름을 직관적으로 변경 (예: 라벨 '3' -> 'label_3_ratio')
    df_aggregated.columns = [f'label_{col}_ratio' for col in df_aggregated.columns]
    
    # 장소별 가장 비율이 높은 대표 감성 라벨(최빈값) 추출
    df_aggregated['top_label'] = df_aggregated.idxmax(axis=1).str.replace('_ratio', '')
    
    # 최종 데이터에 해당 장소의 '총 리뷰 문장 수(sentence_count)' 컬럼 추가 (추천 신뢰도 지표용)
    df_aggregated = df_aggregated.merge(valid_places.set_index(['city', 'place_name']), left_index=True, right_index=True)
    
    # 인덱스 초기화
    df_final = df_aggregated.reset_index()

    print(f"4. 결과를 {output_path}에 저장 중...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
    print("✅ 완료: 통계적 불안정성을 제거한 최종 추천 데이터베이스가 완성되었습니다.")

if __name__ == "__main__":
    # 데이터 경로 (본인의 디렉토리 구조에 맞게 수정하세요)
    INPUT_FILE = "data/sentence_labeled_data.csv" 
    OUTPUT_FILE = "data/final_travel_database.csv" 
    
    # 임계값(min_sentence_count)을 5로 설정하여 함수 실행
    aggregate_place_labels_advanced(INPUT_FILE, OUTPUT_FILE, min_sentence_count=5)