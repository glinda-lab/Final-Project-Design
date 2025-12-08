import streamlit as st
import requests
import openai
import matplotlib.pyplot as plt
import numpy as np
import os
import json 
import io

# --- 1. 환경 설정 및 API 키 설정 ---
st.set_page_config(layout="wide", page_title="AI 기반 생성형 미술 디자이너 (갤러리)")

# OpenAI API 키 설정
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
         st.error("⚠️ OpenAI API 키를 설정해주세요! (Streamlit Secrets 또는 환경 변수)")


# MET Museum API 기본 URL
MET_API_BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"

# 초기 상태 설정 및 키 정리
if 'search_triggered' not in st.session_state:
    st.session_state['search_triggered'] = False
if 'ai_params' not in st.session_state:
    st.session_state['ai_params'] = None
if 'artwork_list' not in st.session_state:
    st.session_state['artwork_list'] = []
if 'point_count_key' not in st.session_state:
    st.session_state['point_count_key'] = 500
if 'selected_artwork_details' not in st.session_state: # 선택된 작품 상세 정보를 저장
    st.session_state['selected_artwork_details'] = None


# --- 2. MET Museum API 함수 ---
@st.cache_data(ttl=3600)
def fetch_artworks(search_term):
    """MET API에서 검색어를 바탕으로 유효한 작품 ID 리스트를 가져옵니다."""
    if not search_term:
        return []
    search_url = f"{MET_API_BASE_URL}/search"
    params = {'q': search_term, 'hasImages': True, 'isPublicDomain': True}
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('objectIDs', [])[:50] 
    except requests.exceptions.RequestException:
        return []

@st.cache_data(ttl=3600)
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

# --- 3. AI 분석 및 디자인 파라미터 추출 함수 ---
def get_ai_design_suggestions(artwork_image_url, artwork_title):
    if not openai.api_key:
        st.error("AI 분석을 위해 OpenAI API 키가 필요합니다.")
        return None

    system_prompt = (
        "당신은 전문 미술 비평가이자 생성형 포스터 디자이너입니다. 제공된 명화 이미지를 분석하여 그 핵심 디자인 요소(색상 팔레트, 주된 레이아웃 특징, 질감/스타일)를 설명하고, "
        "이를 바탕으로 Python Matplotlib 생성형 포스터 코드에 사용할 3가지 핵심 파라미터를 JSON 형식으로 제안하세요. "
        "출력 JSON은 반드시 'analysis' (분석 텍스트), 'color_palette' (4개의 HEX 코드 리스트), 'layers' (3~10 사이 정수), 'wobble_factor' (0.1~0.5 사이 부동소수점) 네 가지 키를 포함해야 합니다. "
        "분석 결과와 JSON만 출력하세요."
    )
    user_prompt = f"이 작품 '{artwork_title}'을 분석하고 디자인 파라미터를 추출해 주세요."

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": artwork_image_url, "detail": "low"}}
                ]}
            ],
            temperature=0.7,
            response_format={"type": "json_object"} 
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        st.error(f"AI 분석 중 오류 발생: {e}")
        return None

# --- 4. 생성형 포스터 생성 함수 (3가지 스타일) ---
def setup_canvas(title):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, color='gray')
    np.random.seed(42)
    return fig, ax

def generate_impressionism_touch_poster(params, point_count):
    """스타일 1: 인상주의 터치"""
    colors = params.get('color_palette', ['#FF0000', '#0000FF', '#00FF00', '#FFFF00'])
    layers = params.get('layers', 5)
    wobble = params.get('wobble_factor', 0.2)
    
    fig, ax = setup_canvas("스타일 1: 인상주의 터치")
    
    N_POINTS = point_count 
    
    for i in range(layers):
        color = colors[i % len(colors)] 
        x = np.random.uniform(0, 1, N_POINTS) + np.random.normal(0, wobble * 0.1) 
        y = np.random.uniform(0, 1, N_POINTS) + np.random.normal(0, wobble * 0.1)
        
        ax.scatter(x, y, s=np.random.uniform(10, 50), color=color, alpha=0.15, edgecolors='none') 
    return fig

def generate_layered_lines_poster(params, point_count):
    """스타일 2: 레이어드 라인"""
    colors = params.get('color_palette', ['#FF0000', '#0000FF', '#00FF00', '#FFFF00'])
    layers = params.get('layers', 5)
    wobble = params.get('wobble_factor', 0.2)
    
    fig, ax = setup_canvas("스타일 2: 레이어드 라인")
    
    N_LINES = point_count 
    
    for i in range(N_LINES):
        color = colors[i % len(colors)]
        start = np.random.uniform(0, 1, 2)
        end = np.random.uniform(0, 1, 2)
        
        if i % 2 == 0: 
             ax.plot([start[0], end[0] + wobble*0.5], 
                     [start[1] + np.random.normal(0, wobble*0.05), start[1] + np.random.normal(0, wobble*0.05)], 
                    color=color, linewidth=np.random.uniform(1, 5), alpha=0.3, zorder=i)
        else: 
             ax.plot([start[0] + np.random.normal(0, wobble*0.05), start[0] + np.random.normal(0, wobble*0.05)], 
                     [start[1], end[1] + wobble*0.5], 
                    color=color, linewidth=np.random.uniform(1, 5), alpha=0.3, zorder=i)
    return fig

def generate_convex_tiles_poster(params):
    """스타일 3: 볼록한 타일"""
    colors = params.get('color_palette', ['#FF0000', '#0000FF', '#00FF00', '#FFFF00'])
    layers = params.get('layers', 5)
    wobble = params.get('wobble_factor', 0.2)
    
    fig, ax = setup_canvas("스타일 3: 볼록한 타일")
    
    GRID_SIZE = layers
    STEP = 1.0 / GRID_SIZE
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            color = colors[(i * GRID_SIZE + j) % len(colors)]
            center_x = i * STEP + STEP / 2
            center_y = j * STEP + STEP / 2
            radius = (STEP / 2) * (1 - wobble * np.random.rand())
            
            circle = plt.Circle((center_x, center_y), radius, color=color, alpha=0.8, edgecolor='none')
            ax.add_patch(circle)
    return fig

# --- 5. Streamlit 메인 앱 구현 ---
def main():
    st.title("🖼️ AI 기반 생성형 미술 디자이너")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["🖼️ 작품 분석 및 포스터 생성", "💡 확장 가이드"])

    with st.sidebar:
        st.header("설정 및 검색")
        
        # 1. 명화 검색 UI
        search_query = st.text_input("🖼️ MET 박물관 작품 검색", st.session_state.get('last_query', "Monet"))
        st.session_state['last_query'] = search_query

        # --- 검색 버튼 ---
        if st.button("🔍 검색 실행", type="secondary"):
            st.session_state['search_triggered'] = True
            st.session_state['ai_params'] = None # 초기화
            st.session_state['artwork_list'] = [] # 초기화
            st.session_state['selected_artwork_details'] = None # 선택 작품 초기화
            
            with st.spinner(f"'{search_query}' 작품 ID 검색 중..."):
                object_ids = fetch_artworks(search_query)
            
            if object_ids:
                temp_list = []
                for obj_id in object_ids:
                    detail = get_artwork_details(obj_id)
                    if detail and detail['image_url']:
                        temp_list.append(detail)
                st.session_state['artwork_list'] = temp_list
                
            if not st.session_state['artwork_list']:
                st.warning("검색 결과가 없거나 이미지가 포함된 작품이 없습니다.")
                st.session_state['search_triggered'] = False

        st.markdown("---")
        st.header("포스터 미세 조정")
        # 점/선 개수 입력 슬라이더
        st.slider(
            '점/선 개수 (밀도)', 
            100, 
            2000, 
            st.session_state['point_count_key'],
            100, 
            key='point_count_key',
            help="인상주의 터치 및 레이어드 라인 스타일에서 사용되는 요소의 개수를 조절합니다."
        )


    with tab1:
        # 💡 세션 상태에서 선택된 작품을 가져옵니다.
        selected_artwork = st.session_state.get('selected_artwork_details')
        point_count_val = st.session_state.get('point_count_key', 500)
        
        if selected_artwork:
            # --- 2. 작품 상세 정보 및 AI 분석 UI (선택 완료 시) ---
            st.header(f"🖼️ 원본 작품: {selected_artwork['title']}")
            st.markdown(f"**작가:** {selected_artwork['artist']} | **ID:** {selected_artwork['object_id']}")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(selected_artwork['image_url'], use_column_width=True, caption=selected_artwork['title'])
                
            with col2:
                st.subheader("작품 분석 및 포스터 생성")
                
                if st.button("🤖 AI 분석 및 디자인 파라미터 추출 시작", type="primary"):
                    st.session_state['ai_params'] = None 
                    with st.spinner("AI가 명화 분석 및 파라미터 추출 중입니다..."):
                        params = get_ai_design_suggestions(selected_artwork['image_url'], selected_artwork['title'])
                        st.session_state['ai_params'] = params
                
                if st.session_state['ai_params']:
                    params = st.session_state['ai_params']
                    
                    st.markdown("---")
                    st.subheader("📝 AI의 디자인 분석 및 제안")
                    analysis_text = params.get('analysis', "분석 결과가 없습니다.")
                    st.info(analysis_text)
                    
                    st.markdown("### 📐 추출된 생성형 파라미터")
                    param_display = {k: v for k, v in params.items() if k != 'analysis'}
                    st.code(json.dumps(param_display, indent=2))
                    
                    st.markdown("---")
                    st.subheader("✨ 생성형 포스터 결과")

                    selected_style = st.selectbox(
                        "🎨 포스터 스타일 선택", 
                        ["인상주의 터치", "레이어드 라인", "볼록한 타일"]
                    )
                    
                    poster_fig = None
                    try:
                        if selected_style == "레이어드 라인":
                            poster_fig = generate_layered_lines_poster(st.session_state['ai_params'], point_count_val)
                        elif selected_style == "볼록한 타일":
                            poster_fig = generate_convex_tiles_poster(st.session_state['ai_params']) 
                        else:
                            poster_fig = generate_impressionism_touch_poster(st.session_state['ai_params'], point_count_val)
                        
                        st.pyplot(poster_fig)
                        st.success(f"포스터 생성 완료! (스타일: {selected_style})")
                        
                        buf = io.BytesIO()
                        poster_fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1)
                        
                        st.download_button(
                            label="💾 포스터 PNG 다운로드",
                            data=buf.getvalue(),
                            file_name=f"{selected_artwork['title']}_{selected_style}_poster.png",
                            mime="image/png"
                        )
                        
                    except Exception as e:
                        st.error(f"포스터 생성 중 오류 발생: {e}")
                        
        else:
            # --- 3. 갤러리 형식 검색 결과 표시 (작품 선택 전) ---
            if st.session_state.get('search_triggered') and st.session_state['artwork_list']:
                st.header("🔍 검색 결과 갤러리")
                artwork_details_list = st.session_state['artwork_list']
                
                cols = st.columns(3) 
                
                for index, art in enumerate(artwork_details_list):
                    col = cols[index % 3] 
                    
                    with col:
                        st.image(art['image_url'], use_column_width=True)
                        st.caption(f"**{art['title']}** - {art['artist']}")
                        
                        # 버튼 클릭 시 세션 상태 업데이트 후 재실행
                        if st.button("이 작품 선택", key=f"select_art_{art['object_id']}"):
                            st.session_state['selected_artwork_details'] = art
                            st.experimental_rerun() 
                            
                st.markdown("---")
                st.info("갤러리에서 '이 작품 선택' 버튼을 눌러 분석을 시작하세요.")
            
            elif st.session_state.get('search_triggered') and not st.session_state['artwork_list']:
                 st.warning("⚠️ 검색 결과가 없거나 이미지가 포함된 작품이 없습니다. 다른 검색어를 시도해 보세요.")
            
            else:
                 st.info("검색어를 입력하고 '검색 실행' 버튼을 눌러 프로젝트를 시작하세요.")

    with tab2:
        st.header("💡 추가 확장 및 배포 가이드")
        st.markdown("""
        ### 1. 갤러리 기능 추가
        - 현재 다운로드 버튼을 통해 개별 파일을 얻을 수 있지만, 생성된 포스터 정보를 세션 상태에 저장하여 별도의 '갤러리' 탭에 모아 볼 수 있습니다.

        ### 2. 최종 배포
        - Github 저장소에 `streamlit_app.py`, `requirements.txt`를 커밋하고 Streamlit Cloud에 배포합니다.
        """)


if __name__ == "__main__":
    main()
