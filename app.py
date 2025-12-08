import streamlit as st
import requests
import openai
import matplotlib.pyplot as plt
import numpy as np
import os
import json # AI 분석 결과를 JSON으로 파싱하기 위해 사용

# --- 1. 환경 설정 및 API 키 설정 ---
st.set_page_config(layout="wide", page_title="AI 생성형 미술 디자이너")

# OpenAI API 키 설정
try:
    # Streamlit Cloud 환경에서 secrets에서 키 불러오기
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    # 로컬 환경 변수에서 불러오기 (개발 시)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
         st.error("⚠️ OpenAI API 키를 설정해주세요! (Streamlit Secrets 또는 환경 변수)")


# MET Museum API 기본 URL
MET_API_BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"

# --- 2. MET Museum API 함수 (2단계) ---

def fetch_artworks(search_term):
    """MET API에서 검색어를 바탕으로 유효한 작품 ID 리스트를 가져옵니다."""
    if not search_term:
        return []
        
    search_url = f"{MET_API_BASE_URL}/search"
    params = {
        'q': search_term,
        'hasImages': True, # 이미지가 있는 작품만
        'isPublicDomain': True # 공개 도메인 작품만
    }
    
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # 상위 50개까지만 가져와 처리 속도 개선
        return data.get('objectIDs', [])[:50] 
    except requests.exceptions.RequestException as e:
        st.error(f"작품 검색 중 오류가 발생했습니다: {e}")
        return []

@st.cache_data(ttl=3600) # 1시간 캐싱하여 API 호출 횟수 줄이기
def get_artwork_details(object_id):
    """특정 작품 ID의 상세 정보(이미지 URL, 제목, 작가)를 가져옵니다."""
    
    detail_url = f"{MET_API_BASE_URL}/objects/{object_id}"
    
    try:
        response = requests.get(detail_url)
        response.raise_for_status()
        details = response.json()
        
        return {
            'title': details.get('title', '제목 없음'),
            'artist': details.get('artistDisplayName', '작가 미상'),
            'image_url': details.get('primaryImageSmall', details.get('primaryImage', '')),
            'object_id': details.get('objectID')
        }
    except requests.exceptions.RequestException:
        return None

# --- 3. AI 분석 및 디자인 파라미터 추출 함수 (3단계) ---

def get_ai_design_suggestions(artwork_image_url, artwork_title):
    """AI에게 작품 이미지와 제목을 주어 디자인 제안을 요청하고 JSON으로 받습니다."""
    
    # AI 역할 프롬프트 정의 (Week 11 역할 기반 챗봇 내용 적용)
    system_prompt = (
        "당신은 전문 미술 비평가이자 생성형 포스터 디자이너입니다. "
        "제공된 명화 이미지를 분석하여 그 핵심 디자인 요소(색상 팔레트, 주된 레이아웃 특징, 질감/스타일)를 설명하고, "
        "이를 바탕으로 Python Matplotlib 생성형 포스터 코드에 사용할 3가지 핵심 파라미터를 JSON 형식으로 제안하세요. "
        "파라미터는 4가지 HEX 색상 코드, 레이어 수(3~10), 불규칙성(0.1~0.5)입니다. "
        "분석 결과와 JSON만 출력하세요."
    )
    
    # 텍스트 요청 내용
    user_prompt = f"이 작품 '{artwork_title}'을 분석하고 디자인 파라미터를 추출해 주세요."

    # GPT-4V를 사용한 멀티모달 호출 구성
    try:
        response = openai.chat.completions.create(
            model="gpt-4o", # 이미지를 처리할 수 있는 모델 (gpt-4o, gpt-4-vision-preview 등)
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": artwork_image_url, "detail": "low"}} # 이미지 URL 전달
                ]}
            ],
            temperature=0.7,
            response_format={"type": "json_object"} # JSON 출력을 요청
        )
        
        # 응답 텍스트를 JSON으로 파싱
        content = response.choices[0].message.content
        ai_data = json.loads(content)
        return ai_data
        
    except openai.APIError as e:
        st.error(f"AI 분석 중 API 오류 발생: {e}")
        return None
    except json.JSONDecodeError:
        st.warning("AI가 유효한 JSON 형식으로 응답하지 못했습니다. 다시 시도해 주세요.")
        return None
    except Exception as e:
        st.error(f"예상치 못한 오류 발생: {e}")
        return None


# --- 4. 생성형 포스터 생성 함수 (4단계) ---
# 이 함수는 3단계에서 추출된 파라미터를 받아서 포스터를 생성합니다. (4단계에서 완성할 예정)
def generate_generative_poster(params):
    """AI가 제안한 파라미터를 사용하여 Matplotlib 포스터를 생성합니다."""
    
    # 파라미터 유효성 검사 및 언팩 (예시 값 사용)
    colors = params.get('color_palette', ['#FF0000', '#0000FF', '#00FF00', '#FFFF00'])
    layers = params.get('layers', 5)
    wobble = params.get('wobble_factor', 0.2)
    
    # Matplotlib을 사용한 생성형 디자인 (Week 3 포스터 개념 적용)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#222222") # 배경색 설정
    ax.set_xticks([])
    ax.set_yticks([])
    
    np.random.seed(42) # 재현성을 위해 시드 사용
    
    for i in range(layers):
        radius = 1.0 - (i / layers) * 0.8
        
        # AI가 제안한 wobble factor를 적용하여 불규칙한 중앙점 계산
        center_x = 0.5 + np.random.uniform(-wobble, wobble)
        center_y = 0.5 + np.random.uniform(-wobble, wobble)
        
        # 색상을 순환하며 적용
        color = colors[i % len(colors)] 
        
        # 단순 원형 패턴 생성 (투명도 조절로 질감 표현)
        circle = plt.Circle((center_x, center_y), radius, color=color, alpha=0.5 / layers, edgecolor='none')
        ax.add_artist(circle)
        
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    return fig


# --- 5. Streamlit 메인 앱 구현 (Main Loop) ---
def main():
    st.title("🖼️ AI 기반 생성형 미술 디자이너")
    st.markdown("---")
    st.sidebar.header("설정 및 검색")
    
    # 1. 명화 검색 및 선택 UI (2단계)
    search_query = st.sidebar.text_input("🖼️ MET 박물관 작품 검색", "Monet")
    artwork_details_list = []
    selected_artwork = None

    if search_query:
        # 작품 ID 목록을 가져옴
        with st.spinner(f"'{search_query}' 검색 중..."):
            object_ids = fetch_artworks(search_query)
        
        if object_ids:
            # 상세 정보 리스트 생성
            for obj_id in object_ids:
                detail = get_artwork_details(obj_id)
                if detail and detail['image_url']:
                    artwork_details_list.append(detail)
                    
            options = [f"{art['title']} - {art['artist']}" for art in artwork_details_list]
            selected_option = st.sidebar.selectbox("🎨 작품 선택", options)
            
            if selected_option:
                selected_artwork = next((art for art in artwork_details_list if f"{art['title']} - {art['artist']}" == selected_option), None)

    # 2. 선택된 작품 표시 및 AI 분석 실행 (3단계)
    if selected_artwork:
        st.header(f"🖼️ 원본 작품: {selected_artwork['title']}")
        st.markdown(f"**작가:** {selected_artwork['artist']} | **ID:** {selected_artwork['object_id']}")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(selected_artwork['image_url'], use_column_width=True, caption=selected_artwork['title'])
            
        with col2:
            st.subheader("작품 분석 및 포스터 생성")
            
            # AI 분석 결과가 저장될 상태 변수
            if 'ai_params' not in st.session_state:
                st.session_state['ai_params'] = None

            # AI 분석 버튼 클릭 시 3단계 실행
            if st.button("🤖 AI 분석 및 디자인 파라미터 추출 시작", type="primary"):
                with st.spinner("AI가 명화 분석 및 파라미터 추출 중입니다..."):
                    params = get_ai_design_suggestions(selected_artwork['image_url'], selected_artwork['title'])
                    st.session_state['ai_params'] = params
            
            # AI 분석 결과가 있을 경우 (3단계 결과 표시)
            if st.session_state['ai_params']:
                params = st.session_state['ai_params']
                
                st.markdown("### 📝 AI의 디자인 분석 및 제안")
                st.info(params.get('analysis', '분석 결과 없음'))
                
                st.markdown("### 📐 추출된 생성형 파라미터")
                st.code(json.dumps({k: v for k, v in params.items() if k != 'analysis'}, indent=2))
                
                st.markdown("---")
                st.subheader("✨ AI 기반 생성형 포스터 결과")
                
                # 4. 생성형 포스터 생성 및 표시 (4단계)
                try:
                    poster_fig = generate_generative_poster(params)
                    st.pyplot(poster_fig)
                    st.success("포스터 생성 완료! 파라미터를 수정하여 다른 포스터를 생성할 수 있습니다.")
                except Exception as e:
                    st.error(f"포스터 생성 중 오류 발생: {e}")
                    
    else:
        st.info("검색어를 입력하고 이미지가 있는 작품을 선택한 후, 'AI 분석' 버튼을 눌러 프로젝트를 시작하세요.")

if __name__ == "__main__":
    main()
