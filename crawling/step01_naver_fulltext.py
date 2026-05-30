import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os

def get_naver_blog_text(url):
    """
    네이버 블로그 URL에 접속하여 iframe 구조를 우회하고 실제 본문 텍스트를 추출하는 함수입니다.
    """
    try:
        # 네이버 서버의 봇 차단을 방지하기 위해 일반 브라우저인 것처럼 User-Agent 헤더를 추가합니다.
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 네이버 블로그는 본문이 iframe(#mainFrame) 안에 숨겨져 있으므로 해당 요소를 찾습니다.
        iframe = soup.select_one("iframe#mainFrame")
        
        # 네이버 블로그(blog.naver.com) 도메인이 아니거나 iframe이 없는 경우(예: 티스토리 등) 중단합니다.
        if not iframe:
            return ""
            
        # iframe의 src 속성을 추출하여 실제 본문이 있는 진짜 URL을 조합합니다.
        iframe_url = "https://blog.naver.com" + iframe.get('src')
        
        # 조합된 진짜 URL로 다시 접속(2차 요청)합니다.
        res_iframe = requests.get(iframe_url, headers=headers, timeout=10)
        soup_iframe = BeautifulSoup(res_iframe.text, 'html.parser')
        
        # 스마트에디터 ONE 방식과 구버전 에디터 방식의 본문 클래스를 모두 탐색합니다.
        text_area = soup_iframe.select_one('.se-main-container') 
        if not text_area:
            text_area = soup_iframe.select_one('#postViewArea')
            
        if text_area:
            # 본문 영역에서 텍스트만 추출하고, 태그 사이의 불필요한 줄바꿈을 공백으로 치환합니다.
            return text_area.get_text(separator=' ', strip=True)
        else:
            return ""
            
    except Exception as e:
        # 접속 지연 등 에러 발생 시 빈 문자열을 반환하여 스크립트 중단을 방지합니다.
        return ""

if __name__ == "__main__":
    # 1. 앞서 검색 API로 수집 완료한 CSV 파일 로드
    input_path = "data/naver_blog_crawling_all_regions.csv"
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"'{input_path}' 파일을 찾을 수 없습니다.")
        
    df = pd.read_csv(input_path)
    
    print(f"총 {len(df)}개 블로그 URL에 대한 본문 스크래핑을 시작합니다...\n")
    
    # 2. 본문을 담을 리스트 초기화
    full_texts = []
    
    # 3. 데이터프레임을 순회하며 URL 접속 및 본문 추출
    for index, row in df.iterrows():
        link = row['link']
        print(f"[{index + 1}/{len(df)}] 본문 추출 중... ", end="", flush=True)
        
        extracted_text = get_naver_blog_text(link)
        
        if extracted_text:
            print("[완료]")
        else:
            print("[실패 또는 외부블로그]")
            
        full_texts.append(extracted_text)
        
        # 대상 서버에 과부하를 주지 않고 IP 차단을 피하기 위해 1초씩 대기합니다.
        time.sleep(1)
        
    # 4. 추출된 본문을 기존 데이터프레임의 'description' 컬럼에 덮어쓰기
    # (이렇게 해야 이후에 작성해둔 정제/병합 코드를 수정 없이 그대로 사용할 수 있습니다.)
    df['description'] = full_texts
    
    # 5. 본문 추출에 실패하여 텍스트가 빈 문자열인 행은 삭제 처리
    df = df[df['description'] != ""]
    
    # 6. 본문이 채워진 새로운 CSV 파일로 저장
    output_path = "data/naver_blog_fulltext_all_regions.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n최종 본문 수집 완료. 총 {len(df)}개의 데이터가 '{output_path}'에 저장되었습니다.")