import pandas as pd
import os
from kiwipiepy import Kiwi

# Kiwi 형태소 분석기 초기화
kiwi = Kiwi()

def extract_target_context(text, target_place, aliases_val, window_before=1, window_after=2):
    """
    하나의 긴 글 안에서 공식 장소명 및 결합된 축약어(aliases)가 언급된 문장을 찾아 
    주변 문맥을 윈도우 범위만큼 추출하는 핵심 함수입니다.
    """
    if not isinstance(text, str) or text.strip() == "":
        return []
    
    try:
        # Kiwi를 통해 텍스트를 문장 단위로 분할
        sentences = kiwi.split_into_sents(text)
        sentence_texts = [s.text for s in sentences if len(s.text.strip()) > 5]
        
        if not sentence_texts:
            return []

        # 검색 키워드 리스트 초기화 (기본적으로 공식 장소명의 공백과 대소문자를 제거하여 추가)
        search_keywords = [str(target_place).replace(" ", "").lower()]
        
        # 엑셀의 aliases 열에 데이터가 존재할 경우 (쉼표 분리 처리)
        if pd.notna(aliases_val) and str(aliases_val).strip() != "":
            alias_list = [a.strip().replace(" ", "").lower() for a in str(aliases_val).split(',')]
            search_keywords.extend(alias_list)
            
        target_indices = []
        
        # 문장들을 순회하며 공식 명칭 또는 축약어가 포함되어 있는지 검사
        for idx, sent in enumerate(sentence_texts):
            normalized_sent = sent.replace(" ", "").lower()
            if any(keyword in normalized_sent for keyword in search_keywords):
                target_indices.append(idx)
        
        # 매칭되는 키워드가 없다면 빈 리스트 반환하여 해당 블로그의 노이즈 차단
        if not target_indices:
            return []

        # 발견된 위치를 기준으로 앞뒤 문맥 추출
        extracted_sentences = []
        for idx in target_indices:
            start_idx = max(0, idx - window_before)
            end_idx = min(len(sentence_texts), idx + window_after + 1)
            
            for i in range(start_idx, end_idx):
                extracted_sentences.append(sentence_texts[i])
        
        # 중복 추출된 문장 제거 후 반환
        return list(dict.fromkeys(extracted_sentences))

    except Exception as e:
        return []

def process_context_segmentation_pipeline():
    # 경로 설정 (경로 에러 방지를 위한 생 데이터 분리)
    data_input_path = "data/final_cleaned_merged_data.csv"
    alias_excel_path = "data/region_JP_aliases.xlsx"
    output_path = "data/sentence_segmented_data.csv"

    if not os.path.exists(data_input_path):
        print(f"[에러] 크롤링 통합 파일을 찾을 수 없습니다: {data_input_path}")
        return
    if not os.path.exists(alias_excel_path):
        print(f"[에러] 축약어 엑셀 파일을 찾을 수 없습니다: {alias_excel_path}")
        return

    print("1. 크롤링 통합 데이터 및 축약어 엑셀 데이터 로드 중...")
    df_data = pd.read_csv(data_input_path)
    df_alias = pd.read_excel(alias_excel_path)

    print("2. 'place_name' 기준 데이터 정렬 및 결합(Merge) 진행 중...")
    # 엑셀 파일의 의도치 않은 공백 제거를 위한 전처리
    df_alias['place_name'] = df_alias['place_name'].astype(str).str.strip()
    df_data['place_name'] = df_data['place_name'].astype(str).str.strip()

    # 원본 데이터에 aliases 열을 Left Join 형태로 결합
    df_merged = pd.merge(df_data, df_alias[['place_name', 'aliases']], on='place_name', how='left')

    print("3. Kiwi 형태소 분석 기반 장소별 핵심 문맥 추출 시작...")
    # row 단위로 가동하기 위해 apply 연산 수행
    df_merged['sentence_list'] = df_merged.apply(
        lambda row: extract_target_context(row['clean_text'], row['place_name'], row['aliases']), 
        axis=1
    )
    
    print("4. 추출 완료 데이터 행 분할 및 구조화 중...")
    # 리스트 데이터를 단일 문장 행들로 전개
    df_exploded = df_merged.explode('sentence_list')
    df_exploded = df_exploded.rename(columns={'sentence_list': 'sentence'})
    
    # 더 이상 파이프라인 후속 단계에서 필요 없는 원본 거대 글 컬럼과 별명 컬럼 제거
    df_exploded = df_exploded.drop(columns=['clean_text', 'aliases'])
    df_exploded = df_exploded.dropna(subset=['sentence'])
    
    # 최종 저장
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df_exploded.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("-" * 30)
    print(f"작업 성공. 장소 연관 문맥 데이터셋이 '{output_path}'에 저장되었습니다.")

if __name__ == "__main__":
    process_context_segmentation_pipeline()