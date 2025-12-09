import streamlit as st
import requests
import openai
import matplotlib.pyplot as plt
import numpy as np
import os
import json 
import io

# --- 1. Environment Setup and API Key Configuration ---
# Setting the wide layout for better visual display
st.set_page_config(layout="wide", page_title="From Canvas To Code: AI Generative Classics")

# OpenAI API Key Configuration
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
         st.error("⚠️ Please set up the OpenAI API Key! (Streamlit Secrets or environment variable)")


# MET Museum API Base URL
MET_API_BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"

# Initialize Session State Variables
if 'search_triggered' not in st.session_state:
    st.session_state['search_triggered'] = False
if 'ai_params' not in st.session_state:
    st.session_state['ai_params'] = None
if 'artwork_list' not in st.session_state:
    st.session_state['artwork_list'] = []
if 'point_count_key' not in st.session_state:
    st.session_state['point_count_key'] = 500
if 'selected_artwork_details' not in st.session_state:
    st.session_state['selected_artwork_details'] = None
# List to store generated poster information
if 'generated_posters' not in st.session_state:
    st.session_state['generated_posters'] = []


# --- 2. MET Museum API Functions ---
@st.cache_data(ttl=3600)
def fetch_artworks(search_term):
    """Fetches a list of valid artwork IDs based on the search term from the MET API."""
    if not search_term:
        return []
    search_url = f"{MET_API_BASE_URL}/search"
    # Use broad search, request images, and public domain items
    params = {'q': search_term, 'hasImages': True, 'isPublicDomain': True}
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        # Fetch only up to 100 IDs to reduce filtering load
        return data.get('objectIDs', [])[:100] 
    except requests.exceptions.RequestException:
        return []

@st.cache_data(ttl=3600)
def get_artwork_details(object_id):
    """Fetches detailed information (image URL, title, artist) for a specific artwork ID."""
    detail_url = f"{MET_API_BASE_URL}/objects/{object_id}"
    try:
        response = requests.get(detail_url)
        response.raise_for_status()
        details = response.json()
        return {
            'title': details.get('title', 'Untitled'),
            'artist': details.get('artistDisplayName', 'Unknown Artist'),
            'image_url': details.get('primaryImageSmall', details.get('primaryImage', '')),
            'object_id': details.get('objectID')
        }
    except requests.exceptions.RequestException:
        # Return None on error
        return None

# --- 3. AI Analysis and Design Parameter Extraction Function ---
def get_ai_design_suggestions(artwork_image_url, artwork_title):
    if not openai.api_key:
        return None

    system_prompt = (
        "You are an expert art critic and generative poster designer. Analyze the provided masterpiece image and describe its core design elements (color palette, main layout features, texture/style). "
        "Based on this, suggest 3 core parameters for use in a Python Matplotlib generative poster code in JSON format. "
        "The output JSON must strictly include the four keys: 'analysis' (analysis text), 'color_palette' (list of 4 HEX codes), 'layers' (integer between 3 and 10), and 'wobble_factor' (float between 0.1 and 0.5). "
        "Output only the analysis and the JSON."
    )
    user_prompt = f"Analyze this artwork '{artwork_title}' and extract design parameters."

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
    except openai.APIError as e:
        # Use st.error for API error feedback
        st.error(f"API Error during AI analysis: Error code: {e.status} - {{'error': '{e.message}'}}")
        st.warning("Please check if the API quota is exceeded or the key has expired.")
        return None
    except Exception as e:
        st.error(f"Error occurred during AI analysis: {e}")
        return None

# --- 4. Generative Poster Creation Functions (3 Styles) ---
def setup_canvas(title):
    # Maintain 8x8 size for gallery consistency
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, color='gray')
    np.random.seed(42)
    return fig, ax

def generate_impressionism_touch_poster(params, point_count):
    """Style 1: Impressionism Touch - Uses AI-suggested color palette."""
    
    # Use the AI-suggested color palette
    colors = params.get('color_palette', ['#FF0000', '#0000FF', '#00FF00', '#FFFF00'])
    
    layers = params.get('layers', 5)
    wobble = params.get('wobble_factor', 0.2)
    
    fig, ax = setup_canvas("Style 1: Impressionism Touch")
    
    N_POINTS = point_count 
    
    for i in range(layers):
        color = colors[i % len(colors)] 
        x = np.random.uniform(0, 1, N_POINTS) + np.random.normal(0, wobble * 0.1) 
        y = np.random.uniform(0, 1, N_POINTS) + np.random.normal(0, wobble * 0.1)
        
        ax.scatter(x, y, s=np.random.uniform(10, 50), color=color, alpha=0.15, edgecolors='none') 
    return fig

def generate_layered_lines_poster(params, point_count):
    """Style 2: Layered Lines - Uses AI-suggested color palette."""
    
    # Use the AI-suggested color palette
    colors = params.get('color_palette', ['#FF0000', '#0000FF', '#00FF00', '#FFFF00'])
    
    layers = params.get('layers', 5)
    wobble = params.get('wobble_factor', 0.2)
    
    fig, ax = setup_canvas("Style 2: Layered Lines")
    
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
    """Style 3: Convex Tiles - Uses AI-suggested color palette."""
    
    # Use the AI-suggested color palette
    colors = params.get('color_palette', ['#FF0000', '#0000FF', '#00FF00', '#FFFF00'])
    
    layers = params.get('layers', 5)
    wobble = params.get('wobble_factor', 0.2)
    
    fig, ax = setup_canvas("Style 3: Convex Tiles")
    
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

# --- 5. Streamlit Main App Implementation ---
def main():
    
    # [Design Improvement] Inject CSS to fix button text color visibility
    st.markdown("""
    <style>
    /* Custom style for main H1 title */
    .title-text {
        font-size: 2.5em; 
        text-align: center;
        font-weight: bold;
    }
    
    /* FIX: Custom style for the specific AI Analysis Button text */
    /* Forces the text inside standard buttons to a dark color for visibility */
    .stButton button p { 
        color: #112250 !important; /* ROYAL BLUE: Dark color for contrast */
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='title-text'>✨ Abstract Classics: AI's New Palette </h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Tab Names
    tab1, tab2 = st.tabs(["🖼️ Artwork Analysis & Poster Generation", "🎨 Poster Gallery"])

    with st.sidebar:
        # Sidebar is dedicated to settings and input only
        st.header("Settings & Search")
        
        # 1. Artwork Search UI
        search_query = st.text_input("✨ MET Museum Artwork Search (Keyword or Term)", st.session_state.get('last_query', "Monet"))
        st.session_state['last_query'] = search_query

        # --- Search Button ---
        if st.button("🔍 Search", type="secondary"):
            st.session_state['search_triggered'] = True
            st.session_state['ai_params'] = None 
            st.session_state['artwork_list'] = [] 
            st.session_state['selected_artwork_details'] = None 
            
            with st.spinner(f"Searching for Artwork IDs and Filtering for '{search_query}'..."):
                object_ids = fetch_artworks(search_query)
            
            if object_ids:
                temp_list = []
                
                for obj_id in object_ids:
                    detail = get_artwork_details(obj_id)
                    
                    if detail is None:
                        continue 
                        
                    # Relaxed Filtering: Only check if a displayable image URL exists.
                    if detail['image_url']:
                        temp_list.append(detail)
                        
                        # Limit results to max 18 for gallery display
                        if len(temp_list) >= 18: 
                             break
                             
                st.session_state['artwork_list'] = temp_list
                
            if not st.session_state['artwork_list']:
                # Use st.warning for feedback
                st.warning("⚠️ No search results found or no images available for the artworks. Check the spelling or try a different search term.")
                st.session_state['search_triggered'] = False

        st.markdown("---")
        st.header("Poster Fine-Tuning")
        # Point/Line Count Slider
        st.slider(
            'Point/Line Count (Density)', 
            100, 
            2000, 
            st.session_state['point_count_key'],
            100, 
            key='point_count_key',
            help="Adjusts the number of elements used in the Impressionism Touch and Layered Lines styles."
        )


    with tab1:
        selected_artwork = st.session_state.get('selected_artwork_details')
        point_count_val = st.session_state.get('point_count_key', 500)
        
        if selected_artwork:
            # --- 2. Artwork Details and AI Analysis UI (After Selection) ---
            st.header(f"🖼️ Original Artwork: {selected_artwork['title']}")
            st.markdown(f"**Artist:** {selected_artwork['artist']} | **ID:** {selected_artwork['object_id']}")
            
            # Use st.columns for visual balance
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(selected_artwork['image_url'], use_column_width=True, caption=selected_artwork['title'])
                
            with col2:
                st.subheader("Artwork Analysis and Poster Generation")
                
                if st.button("🤖 Start AI Analysis and Parameter Extraction", type="primary"):
                    st.session_state['ai_params'] = None 
                    with st.spinner("AI is analyzing the masterpiece and extracting parameters..."):
                        params = get_ai_design_suggestions(selected_artwork['image_url'], selected_artwork['title'])
                        st.session_state['ai_params'] = params
                
                if st.session_state['ai_params']:
                    params = st.session_state['ai_params']
                    
                    st.markdown("---")
                    
                    # Use st.container to group related sections
                    with st.container(border=True):
                        st.subheader("📝 AI Design Analysis and Suggestion")
                        analysis_text = params.get('analysis', "No analysis result available.")
                        
                        # Use st.expander to keep the main screen clean
                        with st.expander("📝 View Detailed AI Analysis"):
                            # Use st.info to display analysis information
                            st.info(analysis_text) 

                        st.markdown("---") 
                        st.markdown("### 📐 Extracted Generative Parameters")
                        
                        param_display = {k: v for k, v in params.items() if k != 'analysis'}
                        
                        # Use st.expander to hide the code parameters
                        with st.expander("⚙️ View Code Parameters"):
                            st.code(json.dumps(param_display, indent=2))
                        
                    st.markdown("---")
                    st.subheader("✨ Generative Poster Result")

                    selected_style = st.selectbox(
                        "🎨 Select Poster Style", 
                        ["Impressionism Touch", "Layered Lines", "Convex Tiles"]
                    )
                    
                    poster_fig = None
                    try:
                        if selected_style == "Layered Lines":
                            poster_fig = generate_layered_lines_poster(st.session_state['ai_params'], point_count_val)
                        elif selected_style == "Convex Tiles":
                            poster_fig = generate_convex_tiles_poster(st.session_state['ai_params']) 
                        else: # Default to Impressionism Touch
                            poster_fig = generate_impressionism_touch_poster(st.session_state['ai_params'], point_count_val)
                        
                        st.pyplot(poster_fig)
                        # Use st.success for positive feedback
                        st.success(f"Poster Generation Complete! (Style: {selected_style})")
                        
                        buf = io.BytesIO()
                        poster_fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1)
                        
                        poster_info = {
                            'title': selected_artwork['title'],
                            'artist': selected_artwork['artist'],
                            'style': selected_style,
                            'image_data': buf.getvalue() 
                        }
                        
                        is_duplicate = any(
                            p['title'] == poster_info['title'] and 
                            p['style'] == poster_info['style'] 
                            for p in st.session_state['generated_posters']
                        )
                        if not is_duplicate:
                            st.session_state['generated_posters'].append(poster_info)


                        st.download_button(
                            label="💾 Download Poster PNG",
                            data=buf.getvalue(),
                            file_name=f"{selected_artwork['title']}_{selected_style}_poster.png",
                            mime="image/png"
                        )
                        
                    except Exception as e:
                        st.error(f"Error occurred during poster generation: {e}")
                        
        else:
            # --- 3. Gallery Display of Search Results (Before Selection) ---
            if st.session_state.get('search_triggered') and st.session_state['artwork_list']:
                st.header("🔍 Search Results Gallery")
                st.caption(f"Artworks broadly related to '**{st.session_state['last_query']}**' have been filtered for images. Click the 'Select This Artwork' button to start analysis.")
                
                artwork_details_list = st.session_state['artwork_list']
                
                # Use columns for a clean grid view
                cols = st.columns(3) 
                
                for index, art in enumerate(artwork_details_list):
                    col = cols[index % 3] 
                    
                    with col:
                        st.image(art['image_url'], use_column_width=True)
                        st.caption(f"**{art['title']}** - {art['artist']}")
                        
                        if st.button("Select This Artwork", key=f"select_art_{art['object_id']}"):
                            st.session_state['selected_artwork_details'] = art
                            st.experimental_rerun() 
                            
            elif st.session_state.get('search_triggered') and not st.session_state['artwork_list']:
                 # Use st.warning for feedback
                 st.warning("⚠️ No search results found or no images available for the artworks. Check the spelling or try a different search term.")
            
            else:
                 st.info("Enter a search query and click 'Execute Search' to start the project.")

    with tab2:
        st.header("🎨 Saved Poster Gallery")
        
        saved_posters = st.session_state['generated_posters']
        
        if not saved_posters:
            st.info("No posters saved yet. Create a poster in the 'Artwork Analysis & Poster Generation' tab and check this gallery.")
        else:
            # Display saved posters in a 3-column gallery format
            num_cols = 3
            cols = st.columns(num_cols)
            
            for index, poster in enumerate(saved_posters):
                col = cols[index % num_cols]
                
                with col:
                    # Display the image from stored byte data
                    # Matplotlib canvas size is fixed (8x8) to maintain aspect ratio consistency
                    col.image(poster['image_data'], use_column_width='always')
                    
                    # Use st.caption with bold text and emoji for clarity
                    col.caption(f"**Original:** {poster['title']} | **Style:** {poster['style']} ✨")
                    
                    # Re-enable download button
                    col.download_button(
                        label="Download",
                        data=poster['image_data'],
                        file_name=f"{poster['title']}_{poster['style']}_saved.png",
                        mime="image/png",
                        key=f"download_saved_{index}"
                    )


if __name__ == "__main__":
    main()
