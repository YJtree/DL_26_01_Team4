import urllib.request
import json
import pandas as pd
import os
import time
import re
from dotenv import load_dotenv

# 1. 환경 변수 로드
load_dotenv()
client_id = os.environ.get("NAVER_CLIENT_ID")
client_secret = os.environ.get("NAVER_CLIENT_SECRET")

if not client_id or not client_secret:
    raise ValueError(".env 파일에 API 키가 설정되지 않음.")

# 2. 엑셀 파일(.xlsx) 로드 (경로 및 확장자 복구)
input_excel_path = r"C:\Users\yjlee\Desktop\YOGIJOGI_JP\crawling\region_JP.xlsx" 
df_places = pd.read_excel(input_excel_path)

all_items = []
clean_html = re.compile('<.*?>')

print(f"총 {len(df_places)}개의 장소에 대한 크롤링을 시작합니다...\n")

# 3. 데이터프레임 순회 및 API 호출
for index, row in df_places.iterrows():
    city = row['city']             
    place_name = row['place_name'] 
    combined_keyword = f"{city} {place_name}"
    
    # 터미널 UI: 줄바꿈 없이 "수집 중..." 출력 대기
    print(f"[{index + 1}/{len(df_places)}] '{combined_keyword}' 수집 중... ", end="", flush=True)
    
    query = urllib.parse.quote(combined_keyword)
    url = f"https://openapi.naver.com/v1/search/blog?query={query}&display=10&start=1"
    
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    
    try:
        response = urllib.request.urlopen(request)
        rescode = response.getcode()
        
        if rescode == 200:
            response_body = response.read()
            data = json.loads(response_body.decode('utf-8'))
            items = data.get('items', [])
            
            num_results = len(items)
            
            # API 응답 결과 개수를 즉시 출력
            if num_results == 0:
                print("❌ 0개 (검색결과 없음)")
            elif num_results <= 2:
                print(f"⚠️ {num_results}개 (결과 부족)")
            else:
                print(f"✅ {num_results}개 완료")
            
            # 데이터 저장 로직
            for item in items:
                clean_title = re.sub(clean_html, '', item['title']).replace('&quot;', '"')
                clean_desc = re.sub(clean_html, '', item['description']).replace('&quot;', '"')
                
                all_items.append({
                    'city': city,                 
                    'place_name': place_name,     
                    'title': clean_title,
                    'link': item['link'],
                    'description': clean_desc,
                    'postdate': item['postdate']
                })
        else:
            print(f"❌ Error Code: {rescode}")
            
    except Exception as e:
        print(f"❌ API 에러 발생: {e}")
        
    time.sleep(0.5)

# 4. 데이터 저장 처리
os.makedirs('data', exist_ok=True)
df_results = pd.DataFrame(all_items)

if not df_results.empty:
    df_results.drop_duplicates(subset=['link'], inplace=True)
    save_path = "data/naver_blog_crawling_all_regions.csv"
    df_results.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n최종 크롤링 완료. 총 {len(df_results)}개의 데이터가 저장되었습니다.")
else:
    print("\n수집된 데이터가 없습니다.")