import os
import requests
import pandas as pd
import time
from dotenv import load_dotenv

# 1. 환경 변수 로드
load_dotenv()
API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY")

if not API_KEY:
    raise ValueError(".env 파일에 GOOGLE_PLACES_API_KEY가 설정되지 않았습니다.")

def get_place_reviews(place_id):
    """
    구글 Place ID를 사용하여 해당 장소의 한국어 리뷰를 가져옵니다.
    """
    url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=name,reviews&language=ko&key={API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    reviews_list = []
    if data.get('status') == 'OK' and 'reviews' in data.get('result', {}):
        for review in data['result']['reviews']:
            text = review.get('text', '').strip()
            if text:  
                reviews_list.append(text)
    return reviews_list

if __name__ == "__main__":
    # 2. 엑셀 파일(.xlsx) 직접 로드
    input_excel_path = r"C:\Users\yjlee\Desktop\YOGIJOGI_JP\crawling\region_JP.xlsx" 
    df_places = pd.read_excel(input_excel_path)

    all_reviews = []

    print(f"총 {len(df_places)}개 장소의 구글 리뷰 크롤링을 시작합니다...\n")

    # 3. 데이터프레임 순회 및 리뷰 수집
    for index, row in df_places.iterrows():
        city = row['city']
        place_name = row['place_name']
        place_id = row['place_ID'] 
        
        # 터미널 UI: 줄바꿈 없이 "수집 중..." 출력 대기
        print(f"[{index + 1}/{len(df_places)}] '{city} {place_name}' 수집 중... ", end="", flush=True)
        
        # place_id 누락 예외 처리
        if pd.isna(place_id) or str(place_id).strip() == "":
            print("[건너뜀] place_ID 누락")
            continue

        try:
            # 리뷰 API 호출
            reviews = get_place_reviews(place_id)
            num_reviews = len(reviews)
            
            # API 응답 결과 개수를 즉시 출력
            if num_reviews == 0:
                print("❌[결과 없음] 0개")
            else:
                print(f"✅[완료] {num_reviews}개")
            
            # 데이터 저장
            for review_text in reviews:
                all_reviews.append({
                    'city': city,
                    'place_name': place_name,
                    'source': 'Google Maps',
                    'description': review_text
                })
                
        except Exception as e:
            print(f"[에러 발생] {e}")
            
        time.sleep(0.5)
        
    # 4. 수집된 데이터를 저장
    if all_reviews:
        df_results = pd.DataFrame(all_reviews)
        os.makedirs('data', exist_ok=True)
        
        save_path = "data/google_places_reviews_all_regions.csv"
        df_results.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n최종 수집 완료. 총 {len(df_results)}개의 리뷰 데이터가 저장되었습니다.")
    else:
        print("\n수집된 리뷰 데이터가 없습니다.")