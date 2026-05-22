import urllib.request
import json
import pandas as pd
import os
import time
import re
from dotenv import load_dotenv

# .env 파일의 환경 변수를 불러옴
load_dotenv()

# 1. API 설정 (환경 변수에서 키를 안전하게 가져옴)
client_id = os.environ.get("NAVER_CLIENT_ID")
client_secret = os.environ.get("NAVER_CLIENT_SECRET")

# 키가 제대로 불러와졌는지 안전 장치 추가
if not client_id or not client_secret:
    raise ValueError(".env 파일에 API 키가 설정되지 않음.")

# 2. 다중 검색어 설정
search_keywords = [
    "일본 소도시 감성 여행",
    "일본 시골 마을",
    "일본 소도시 카페",
    "일본 조용한 소도시",
    "일본 숨은 명소"
]

# 결과를 담을 빈 리스트
all_items = []

# 정규표현식: HTML 태그 제거용 컴파일
clean_html = re.compile('<.*?>')

# 3. 검색어 및 페이지 순회하며 데이터 수집
for keyword in search_keywords:
    print(f"'{keyword}' 검색어 수집 시작...")
    query = urllib.parse.quote(keyword)
    
    # start=1, 101, 201 ... 901 (네이버 검색 API 최대 한도는 1000개)
    for start_idx in range(1, 1000, 100):
        url = f"https://openapi.naver.com/v1/search/blog?query={query}&display=100&start={start_idx}"
        
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
                
                if not items:
                    break # 더 이상 결과가 없으면 해당 검색어 루프 종료
                
                for item in items:
                    # 정규표현식을 이용해 HTML 태그(<b> 등) 및 불필요한 특수문자 제거
                    clean_title = re.sub(clean_html, '', item['title']).replace('&quot;', '"')
                    clean_desc = re.sub(clean_html, '', item['description']).replace('&quot;', '"')
                    
                    all_items.append({
                        'keyword': keyword, # 어떤 검색어로 수집되었는지 출처 남기기
                        'title': clean_title,
                        'link': item['link'],
                        'description': clean_desc,
                        'postdate': item['postdate']
                    })
            else:
                print(f"Error Code: {rescode}")
                
        except Exception as e:
            print(f"API 요청 중 에러 발생 (start={start_idx}): {e}")
            break
            
        # API 호출 제한(Rate Limit)을 피하기 위해 0.5초 대기
        time.sleep(0.5)

# 4. 수집된 전체 데이터를 판다스 데이터프레임으로 변환 및 중복 제거
df = pd.DataFrame(all_items)

if not df.empty:
    # 동일한 블로그 글이 여러 검색어에 중복 노출될 수 있으므로 링크(link)를 기준으로 중복 제거
    df.drop_duplicates(subset=['link'], inplace=True)
    
    # 5. CSV 파일로 저장
    os.makedirs('data', exist_ok=True)
    save_path = "data/naver_blog_crawling_data_large.csv"
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n최종 크롤링 완료. 중복 제거 후 총 {len(df)}개의 정제된 데이터가 '{save_path}'에 저장되었어.")
else:
    print("\n수집된 데이터가 없어.")