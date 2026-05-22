import pandas as pd
import os

def merge_labeled_data_final():
    original_file = "data/final_integrated_labeling.xlsx"
    new_file = "data/remaining_to_label.xlsx"
    
    if not os.path.exists(original_file) or not os.path.exists(new_file):
        print("파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    print("데이터를 불러오고 메타데이터(title)를 보존하며 병합합니다...")
    
    df_original = pd.read_excel(original_file)
    df_new = pd.read_excel(new_file)

    # 1. 헤더(열 이름) 통일
    # 새 파일(remaining_to_label)의 1행에 적힌 한글 이름들을 
    # 오리지널 파일의 기준(place_name, review_text, label)으로 변환합니다.
    # (주의: 새 엑셀 파일에 적힌 실제 열 이름에 맞게 왼쪽 단어를 수정하세요)
    rename_mapping = {
        'review_text': 'review_text',  
        'label': 'label',            
        'title': 'place_name'      # 만약 블로그제목이라 적혀있어도 place_name으로 통합
    }
    df_new = df_new.rename(columns=rename_mapping)

    # 2. 미작업 데이터(빈칸) 삭제
    df_new_cleaned = df_new.dropna(subset=['label']).copy()
    
    # 3. 보존할 필수 3대 기둥 목록
    essential_columns = ['place_name', 'review_text', 'label'] 
    
    # 4. 3대 기둥만 남기고 필터링
    df_original = df_original[essential_columns]
    df_new_cleaned = df_new_cleaned[essential_columns]

    # 5. 데이터 병합 (Concat)
    df_merged = pd.concat([df_original, df_new_cleaned], ignore_index=True)
    df_merged['label'] = df_merged['label'].astype(int)
    
    # 6. 최종 저장
    output_path = "data/golden_dataset.xlsx"
    df_merged.to_excel(output_path, index=False)
    
    print("-" * 30)
    print(f"기존 데이터: {len(df_original)}개")
    print(f"추가된 데이터: {len(df_new_cleaned)}개")
    print(f"최종 병합된 데이터: {len(df_merged)}개")
    print("보존된 컬럼명:", df_merged.columns.tolist())
    print(f"성공! title이 완벽히 보존된 데이터가 '{output_path}'에 저장되었습니다.")
    print("-" * 30)

if __name__ == "__main__":
    merge_labeled_data_final()