import pandas as pd

# 1. 데이터 로드
df_db = pd.read_csv("data/final_travel_database.csv")
df_alias = pd.read_excel("data/region_JP_aliases.xlsx")

# 2. 병합을 위한 전처리 (공백 제거)
# 두 파일의 장소명 컬럼명이 'place_name'이라고 가정합니다.
df_db['place_name'] = df_db['place_name'].astype(str).str.strip()
df_alias['place_name'] = df_alias['place_name'].astype(str).str.strip()

# 3. 데이터 병합 (Left Join)
# df_db의 모든 장소 정보를 유지하면서, df_alias의 'place' 열만 가져옵니다.
df_merged = pd.merge(
    df_db, 
    df_alias[['place_name', 'place']], # 가져올 열만 선택
    on='place_name', 
    how='left'
)

# 4. 결과 확인 및 저장
print(df_merged.head())
df_merged.to_csv("data/final_travel_database_updated.csv", index=False, encoding='utf-8-sig')
print("병합 완료: final_travel_database_updated.csv에 저장되었습니다.")