import os
import itertools
import requests
import random
import pandas as pd
from dotenv import load_dotenv

# .env 파일에 저장된 환경 변수를 시스템 환경으로 불러옴
load_dotenv()

# 환경 변수에서 구글 API 키를 가져와 변수에 할당
API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY")

# API 키가 정상적으로 로드되지 않았을 경우 프로그램 실행을 중단하고 에러 발생
if not API_KEY:
    raise ValueError("환경 변수에 GOOGLE_PLACES_API_KEY가 설정되지 않았습니다.")

def get_places_list(query):
    """
    1단계 (Text Search): 검색어를 기반으로 구글 맵에서 관련된 장소들의 목록을 가져오는 함수입니다.
    """
    # 검색어(query)를 포함하여 구글 Text Search API 요청 URL 생성 (한국어 결과 요청)
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={query}&language=ko&key={API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    places = []
    # 통신 상태가 정상('OK')일 때만 결과 데이터 처리
    if data.get('status') == 'OK':
        for item in data['results']:
            # 각 장소의 이름과 리뷰 검색에 필수적인 고유 식별자(place_id)를 딕셔너리로 저장
            places.append({
                'name': item['name'],
                'place_id': item['place_id']
            })
    return places

def get_place_reviews(place_id, place_name):
    """
    2단계 (Place Details): 1단계에서 얻은 place_id를 사용하여 특정 장소의 리뷰 텍스트를 가져오는 함수입니다.
    """
    # [중요] fields=name,reviews 파라미터를 명시하여 불필요한 사진, 전화번호 데이터 수신을 막고 과금을 최소화함
    url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=name,reviews&language=ko&key={API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    reviews_list = []
    # 통신이 정상이고, 반환된 데이터 내에 'reviews' 키워드가 존재하는 경우에만 처리
    if data.get('status') == 'OK' and 'reviews' in data['result']:
        for review in data['result']['reviews']:
            # 리뷰 텍스트 앞뒤의 불필요한 공백 제거
            text = review.get('text', '').strip()
            # 텍스트가 비어있지 않은 실제 리뷰 데이터만 리스트에 추가
            if text:
                reviews_list.append({
                    'source': 'Google Places API',
                    'keyword': place_name,
                    'description': text
                })
    return reviews_list

if __name__ == "__main__":
    # 1. 재료 준비 (원하는 단어만 몇 개 던져주면 됨)
    regions = ["시코쿠", "구마모토", "다카마쓰", "가고시마", "이시가키","가와고","다카야마","하코다테"]
    vibes = ["조용한", "숨은", "로컬", "현지인", "분위기 좋은", "힙한", "혼자여행"]
    places = ["명소", "카페", "식당"]

    # 2. 파이썬이 모든 경우의 수를 곱해서 검색어 자동 생성 (7 x 6 x 5 = 210개)
    search_queries = [f"{r} {v} {p}" for r, v, p in itertools.product(regions, vibes, places)]
    
    print(f"총 {len(search_queries)}개의 검색어가 자동 생성되었습니다! (예: {search_queries[0]})")# 🌟 핵심 수정: 전체 210개 중에서 15개를 무작위(랜덤)로 골고루 추출합니다.
    test_queries = random.sample(search_queries, 15) 
    
    all_reviews = []
    
    # 순서대로 자른 리스트가 아니라, 랜덤으로 뽑은 test_queries를 사용합니다.
    for query in test_queries: 
        print(f"\n[{query}] 장소 검색 중...")
        places_list = get_places_list(query)
        
        # 잦은 API 호출로 인한 차단(에러) 방지를 위해 최대 5개 장소만 우선 수집 (조절 가능)
        for place in places_list[:5]: 
            reviews = get_place_reviews(place['place_id'], place['name'])
            all_reviews.extend(reviews)
            
    if all_reviews:
        df = pd.DataFrame(all_reviews)
        os.makedirs('data', exist_ok=True)
        save_path = "data/google_places_reviews.csv"
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n수집 완료. 총 {len(df)}개의 데이터가 '{save_path}'에 저장되었습니다.")
    else:
        print("\n수집된 리뷰가 없습니다.")