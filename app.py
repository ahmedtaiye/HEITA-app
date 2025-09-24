import subprocess
import sys
import os
# Attempt to install missing packages
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

# Check and install plotly if missing
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    print("Plotly not found. Installing...")
    if install_package("plotly==5.15.0"):
        import plotly.express as px
        import plotly.graph_objects as go
        print("Plotly installed successfully")
    else:
        print("Failed to install plotly")
# Now import other packages
import streamlit as st
import pandas as pd
import re
from collections import Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import time
import google.generativeai as genai
import json

# --------------------------
# ⚙️ STREAMLIT PAGE CONFIG MUST BE THE FIRST COMMAND
# --------------------------
st.set_page_config(page_title="AI & Digital Learning Insights", layout="wide", page_icon="🎓")

# --------------------------
# 🛠️ NLTK DATA DOWNLOADS
# --------------------------
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

# --------------------------
# 🎨 CUSTOM STYLING
# --------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2ca02c;
        border-bottom: 2px solid #2ca02c;
        padding-bottom: 0.3rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .viz-chat-message {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------
# 🧹 TEXT CLEANING FUNCTION
# --------------------------
def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.strip().lower()


# --------------------------
# 😊 SENTIMENT ANALYSIS (VADER)
# --------------------------
def get_sentiment_vader(text):
    if not text:
        return 0.0
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']


# --------------------------
# 😢 EKMAN EMOTION CLASSIFICATION
# --------------------------
def get_ekman_emotions(text):
    if not text:
        return "neutral"
    emotion_keywords = {
        'anger': ['angry', 'furious', 'outraged', 'frustrated', 'annoyed', 'hostile', 'irritated', 'mad'],
        'disgust': ['disgusted', 'revulsed', 'sickened', 'repulsed', 'nauseated', 'contempt', 'loathe'],
        'fear': ['afraid', 'scared', 'terrified', 'fearful', 'panicked', 'anxious', 'nervous', 'worried'],
        'joy': ['happy', 'joyful', 'delighted', 'excited', 'pleased', 'content', 'glad', 'ecstatic'],
        'sadness': ['sad', 'depressed', 'unhappy', 'grief', 'sorrow', 'melancholy', 'heartbroken', 'disappointed'],
        'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'startled', 'astounded', 'stunned']
    }
    text_lower = text.lower()
    emotion_scores = {emotion: 0 for emotion in emotion_keywords}
    for emotion, keywords in emotion_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            emotion_scores[emotion] += 1
    if sum(emotion_scores.values()) == 0:
        sentiment = get_sentiment_vader(text)
        if sentiment > 0.3:
            return "joy"
        elif sentiment < -0.3:
            return "sadness"
        else:
            return "neutral"
    return max(emotion_scores.items(), key=lambda x: x[1])[0]


# --------------------------
# 📊 TOPIC MODELING WITH KEYWORD EXTRACTION
# --------------------------
def extract_meaningful_topics(texts, num_topics=5, num_keywords=5):
    if not texts or len(texts) == 0:
        return []
    full_text = " ".join([str(t) for t in texts if t])
    words = re.findall(r'\b[a-z]{3,}\b', full_text.lower())
    stop_words = set(stopwords.words('english'))
    survey_stopwords = {'university', 'education', 'learning', 'student', 'technology',
                        'digital', 'institution', 'experience', 'use', 'using'}
    custom_stopwords = stop_words.union(survey_stopwords)
    words = [word for word in words if word not in custom_stopwords]
    word_freq = Counter(words)
    themes = categorize_words_into_themes(word_freq, num_topics)
    return themes


def categorize_words_into_themes(word_freq, num_themes=5):
    theme_categories = {
        'Teaching & Pedagogy': ['teaching', 'pedagogy', 'instruction', 'curriculum', 'course'],
        'Technology & Tools': ['technology', 'software', 'platform', 'tool', 'system'],
        'Access & Equity': ['access', 'equity', 'inclusion', 'opportunity', 'barrier'],
        'Skills & Development': ['skill', 'development', 'training', 'competency', 'ability'],
        'Assessment & Evaluation': ['assessment', 'evaluation', 'feedback', 'grading', 'measurement'],
        'AI & Automation': ['ai', 'artificial', 'intelligence', 'automation', 'algorithm'],
        'Resources & Support': ['resource', 'support', 'funding', 'infrastructure', 'service']
    }
    themes = {}
    for word, count in word_freq.most_common(50):
        assigned = False
        for theme, keywords in theme_categories.items():
            if any(keyword in word for keyword in keywords):
                if theme not in themes:
                    themes[theme] = []
                themes[theme].append((word, count))
                assigned = True
                break
        if not assigned and len(themes) < num_themes:
            theme_name = f"Theme {len(themes) + 1}"
            themes[theme_name] = [(word, count)]
    return themes


# --- Function to Generate Word Cloud ---
def interactive_word_cloud(text_data, title):
    """Streamlit-native interactive word-cloud (no external pop-ups)."""
    if not text_data:
        st.warning("No text data to create a word cloud.")
        return

    full_text = " ".join([t for t in text_data if t])
    if not full_text:
        st.warning("No valid text to create a word cloud.")
        return

    # ----------------  CONTROLS  ----------------
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        max_words = st.slider("Max words", 50, 400, 200, key=f"{title}_words")
    with col2:
        darkness = st.slider("Background darkness", 0, 100, 0, key=f"{title}_bg")
    with col3:
        max_font = st.slider("Max font size", 50, 350, 150, key=f"{title}_font")
    with col4:
        st.write("")  # alignment spacer
        update = st.button("🔄 Update cloud", key=f"{title}_btn")

    # ----------------  BUILD CLOUD  ----------------
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'university', 'education', 'learning', 'student', 'technology',
                        'digital', 'institution', 'experience', 'use'}
    all_stopwords = stop_words.union(custom_stopwords)

    wc = WordCloud(width=800, height=400,
                   background_color='black' if darkness > 50 else 'white',
                   max_words=max_words,
                   max_font_size=max_font,
                   stopwords=all_stopwords,
                   colormap='tab10').generate(full_text)

    # ----------------  PLOT  ----------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(title, fontsize=20, color='white' if darkness > 50 else 'black')
    ax.axis('off')
    st.pyplot(fig)


# --------------------------
# 🎯 LOAD AND PREPARE DATA
# --------------------------
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    required_cols = [
        "Region", "Country",
        "Position of the respondent to the Survey (Please select only one)",
        "Which category best describes your institution?",
        "Which are the key objectives driving digital innovation of your institution?",
        "In which ways has the use of digital technologies enhanced the learning experience of students?",
        "In which ways has the use of digital technologies negatively impacted the learning experience of students?",
        "What are the main challenges identified by your institution with regard to generative AI and its impact on higher education?",
        "What are the main opportunities identified by your institution with regard to generative AI and its impact on higher education?"
    ]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing column: {col}. Please ensure your file has this exact column name.")
            return None, None
    df = df.rename(columns={
        "Position of the respondent to the Survey (Please select only one)": "Position",
        "Which category best describes your institution?": "Institution_Type",
        "Which are the key objectives driving digital innovation of your institution?": "Objectives",
        "In which ways has the use of digital technologies enhanced the learning experience of students?": "Enhanced_Learning",
        "In which ways has the use of digital technologies negatively impacted the learning experience of students?": "Negative_Impact",
        "What are the main challenges identified by your institution with regard to generative AI and its impact on higher education?": "AI_Challenges",
        "What are the main opportunities identified by your institution with regard to generative AI and its impact on higher education?": "AI_Opportunities"
    })
    text_cols = ["Objectives", "Enhanced_Learning", "Negative_Impact", "AI_Challenges", "AI_Opportunities"]
    for col in text_cols:
        df[col + "_clean"] = df[col].apply(clean_text)
        df[col + "_sentiment"] = df[col + "_clean"].apply(get_sentiment_vader)
        df[col + "_emotion"] = df[col + "_clean"].apply(get_ekman_emotions)
    return df, text_cols


# --------------------------
# 🎨 ENHANCED VISUALIZATION FUNCTIONS
# --------------------------
def validate_dataframe(df):
    if df is None or len(df) == 0:
        return False
    required_cols = [
        "Region", "Country", "Position", "Institution_Type",
        "Objectives", "Enhanced_Learning", "Negative_Impact",
        "AI_Challenges", "AI_Opportunities"
    ]
    for col in required_cols:
        if col not in df.columns:
            return False
    return True


def create_enhanced_sentiment_visualization(sentiments, title):
    if not sentiments or not any(isinstance(s, (int, float)) for s in sentiments):
        return None
    fig = px.histogram(x=sentiments, nbins=20, title=title,
                       labels={"x": "Sentiment Score (-1 = Negative, +1 = Positive)"},
                       color_discrete_sequence=["#636EFA"])
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Neutral", annotation_position="top")
    fig.add_vline(x=0.5, line_dash="dot", line_color="green", annotation_text="Positive", annotation_position="top")
    fig.add_vline(x=-0.5, line_dash="dot", line_color="red", annotation_text="Negative", annotation_position="top")
    fig.update_layout(annotations=[dict(x=0.75, y=0.9, xref="paper", yref="paper",
                                        text="Scores > 0.5 indicate strong positive sentiment",
                                        showarrow=False, bgcolor="rgba(255,255,255,0.8)")])
    return fig


def create_enhanced_emotion_visualization(emotions, title):
    if not emotions:
        return None
    emotion_counts = pd.Series(emotions).value_counts()
    emotion_colors = {
        'joy': '#4CAF50', 'surprise': '#FFC107', 'neutral': '#9E9E9E',
        'sadness': '#2196F3', 'fear': '#673AB7', 'anger': '#F44336', 'disgust': '#795548'
    }
    colors = [emotion_colors.get(emotion, '#9E9E9E') for emotion in emotion_counts.index]
    fig = px.pie(values=emotion_counts.values, names=emotion_counts.index, title=title,
                 color=emotion_counts.index, color_discrete_map=emotion_colors)
    fig.update_layout(annotations=[dict(x=0.5, y=-0.15, xref="paper", yref="paper",
                                        text="Based on Ekman's 6 basic emotions + neutral",
                                        showarrow=False, font=dict(size=10, color="gray"))])
    return fig


def generate_insights_summary(df_filtered, selected_col):
    sentiments = df_filtered[selected_col + "_sentiment"].tolist()
    emotions = df_filtered[selected_col + "_emotion"].tolist()
    if not sentiments or not emotions:
        return
    avg_sentiment = np.mean(sentiments)
    emotion_counts = pd.Series(emotions).value_counts()
    primary_emotion = emotion_counts.index[0] if len(emotion_counts) > 0 else "neutral"
    st.markdown("---")
    st.subheader("💡 Key Insights")
    if avg_sentiment > 0.3:
        st.success(
            f"**Overall Positive Sentiment**: Responses show generally positive feelings (average score: {avg_sentiment:.2f})")
    elif avg_sentiment < -0.3:
        st.error(
            f"**Overall Negative Sentiment**: Responses show generally negative feelings (average score: {avg_sentiment:.2f})")
    else:
        st.info(f"**Mixed Sentiment**: Responses show neutral or mixed feelings (average score: {avg_sentiment:.2f})")
    if primary_emotion == "joy":
        st.success(f"**Primary Emotion: Joy** - Respondents express positive feelings about this topic")
    elif primary_emotion in ["anger", 'disgust', 'sadness']:
        st.error(
            f"**Primary Emotion: {primary_emotion.title()}** - Respondents express negative feelings about this topic")
    elif primary_emotion == "fear":
        st.warning(f"**Primary Emotion: Fear** - Respondents express concerns or anxieties about this topic")
    else:
        st.info(f"**Primary Emotion: {primary_emotion.title()}** - Respondents show varied emotional responses")


# --- Function to create a box plot for comparative analysis ---
def create_comparative_box_plot(df, col1, col2, title):
    """Create a comparative box plot for sentiment scores."""
    df_melted = pd.melt(df, id_vars=[col1, col2], value_vars=[col1 + "_sentiment", col2 + "_sentiment"])
    fig = px.box(df_melted, x="variable", y="value",
                 title=title, color="variable",
                 labels={'variable': 'Topic', 'value': 'Sentiment Score'},
                 color_discrete_map={col1 + "_sentiment": "#1f77b4", col2 + "_sentiment": "#d62728"})
    fig.update_layout(xaxis_title="Topic", yaxis_title="Sentiment Score")
    return fig


# --- New Function to Create Comparative Sunburst Chart (improved for clarity) ---
def create_comparative_sunburst(df, demographic_col, text_col):
    """Generates a sunburst chart to compare a text metric across a demographic."""
    df_grouped = df.groupby([demographic_col, text_col]).size().reset_index(name='count')

    fig = px.sunburst(df_grouped, path=[demographic_col, text_col], values='count',
                      title=f"Breakdown of {text_col.replace('_', ' ')} by {demographic_col}",
                      color=text_col,
                      color_discrete_map={
                          'joy': '#4CAF50', 'surprise': '#FFC107', 'neutral': '#9E9E9E',
                          'sadness': '#2196F3', 'fear': '#673AB7', 'anger': '#F44336',
                          'disgust': '#795548', 'Positive': '#28a745', 'Negative': '#dc3545',
                          'Neutral': '#6c757d'
                      })
    fig.update_layout(height=600)
    return fig


# --- New Function to Create Comparative Top Words Bar Chart ---
def create_top_words_comparison(df, demographic_col, text_col, top_n=10):
    st.subheader(f"Top {top_n} Words by {demographic_col}")
    categories = df[demographic_col].dropna().unique()
    if len(categories) < 2:
        st.warning(f"Not enough unique categories in '{demographic_col}' to perform a comparison.")
        return
    comparison_data = []
    full_text_col = text_col + '_clean'
    for category in categories:
        texts = df[df[demographic_col] == category][full_text_col].tolist()
        if not texts:
            continue
        full_text = " ".join([t for t in texts if t])
        words = re.findall(r'\b[a-z]{3,}\b', full_text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]
        word_counts = Counter(filtered_words)
        for word, count in word_counts.most_common(top_n):
            comparison_data.append({'Demographic': category, 'Word': word, 'Count': count})
    if not comparison_data:
        st.warning("No data to compare top words.")
        return
    comp_df = pd.DataFrame(comparison_data)
    fig = px.bar(comp_df, x='Word', y='Count', color='Demographic',
                 barmode='group', title=f"Top {top_n} Words: Comparative Analysis")
    st.plotly_chart(fig, use_container_width=True)


def add_visualization_help():
    """Add helpful tooltips and explanations for visualizations"""
    with st.expander("ℹ️ How to interpret these visualizations"):
        st.markdown("""
        ### Understanding the Charts

        **Word Cloud**:
        - Larger words appear more frequently in responses
        - Word size indicates importance/frequency

        **Sentiment Analysis**:
        - Scores range from -1 (very negative) to +1 (very positive)
        - Values near 0 indicate neutral sentiment
        - The histogram shows distribution of sentiment scores

        **Emotion Analysis**:
        - Based on Ekman's 6 basic emotions + neutral
        - Pie chart shows proportion of each emotion detected

        **Thematic Analysis**:
        - Groups related words into meaningful themes
        - Bar chart shows frequency of words within each theme
        """)


# --------------------------
# 🤖 VISUALIZATION INTERPRETER CHATBOT (NEW)
# --------------------------
def create_visualization_interpreter(df_filtered, selected_col, sentiment_chart, emotion_chart, themes,
                                     analysis_type="question"):
    """Create a chatbot that interprets visualization results"""

    st.markdown("---")
    st.markdown('<h2 class="section-header">🤖 Visualization Interpreter</h2>', unsafe_allow_html=True)

    # Extract key metrics for context
    sentiments = df_filtered[selected_col + "_sentiment"].tolist()
    emotions = df_filtered[selected_col + "_emotion"].tolist()

    avg_sentiment = np.mean(sentiments) if sentiments else 0
    emotion_counts = pd.Series(emotions).value_counts()
    primary_emotion = emotion_counts.index[0] if len(emotion_counts) > 0 else "neutral"

    # Get top words for word cloud context
    def get_top_words(texts, n=10):
        if not texts:
            return []
        full_text = " ".join([str(t) for t in texts if t])
        words = re.findall(r'\b[a-z]{3,}\b', full_text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(n)]

    top_words = get_top_words(df_filtered[selected_col + "_clean"].tolist())

    # Create context for the AI
    visualization_context = f"""
    VISUALIZATION ANALYSIS CONTEXT:

    ANALYSIS TYPE: {analysis_type.upper()} ANALYSIS
    QUESTION BEING ANALYZED: {selected_col.replace('_', ' ').title()}

    SENTIMENT ANALYSIS:
    - Average sentiment score: {avg_sentiment:.3f}
    - Sentiment distribution: {len([s for s in sentiments if s > 0.3])} positive, 
      {len([s for s in sentiments if s < -0.3])} negative, 
      {len([s for s in sentiments if -0.3 <= s <= 0.3])} neutral

    EMOTION ANALYSIS:
    - Primary emotion: {primary_emotion}
    - Emotion distribution: {dict(emotion_counts)}

    THEMATIC ANALYSIS:
    - Key themes identified: {list(themes.keys()) if themes else 'None'}
    - Top words per theme: { {theme: [word for word, count in words[:3]] for theme, words in themes.items()} if themes else 'None'}

    WORD CLOUD INSIGHTS:
    - Top 10 most frequent words: {top_words}

    DATA CONTEXT:
    - Sample size: {len(df_filtered)} responses
    - Available demographics: Region, Country, Position, Institution Type
    """

    # Initialize chat session for visualization interpreter
    if "viz_chat_history" not in st.session_state:
        st.session_state.viz_chat_history = []

    # Display chat history
    for message in st.session_state.viz_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    viz_prompt = st.chat_input("Ask about what these visualizations mean...")

    if viz_prompt:
        # Add user message to history
        st.session_state.viz_chat_history.append({"role": "user", "content": viz_prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(viz_prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            system_prompt = f"""
            You are a data visualization interpreter specializing in survey analysis. 

            {visualization_context}

            GUIDELINES:
            1. Interpret the visualizations in simple, clear language
            2. Explain what the sentiment scores and emotion distributions mean
            3. Connect the thematic analysis to practical implications
            4. Highlight surprising or important patterns
            5. Suggest what actions might be taken based on these insights
            6. Be concise but informative (2-3 paragraphs maximum)
            7. Reference specific numbers and patterns from the data
            8. Explain why certain emotions or sentiments might be dominant

            User's question: {viz_prompt}

            Provide a helpful interpretation:
            """

            try:
                # Use the existing model instance
                response = model.generate_content(system_prompt)

                full_response = response.text
                st.markdown(full_response)
                st.session_state.viz_chat_history.append({"role": "assistant", "content": full_response})

            except Exception as e:
                error_msg = f"Error generating interpretation: {str(e)}"
                st.error(error_msg)
                st.session_state.viz_chat_history.append({"role": "assistant", "content": error_msg})

    # Add example questions
    with st.expander("💡 Example questions to ask"):
        st.markdown("""
        **General Interpretation Questions:**
        - What do these sentiment results tell us?
        - Why is [emotion] the dominant emotion here?
        - What practical implications can we draw from these themes?
        - Are there any surprising patterns in this data?
        - How should we act on these insights?

        **Specific Analysis Questions:**
        - Explain the sentiment distribution pattern
        - What do the most frequent words indicate about priorities?
        - How do the emotions relate to the sentiment scores?
        - What do the thematic clusters suggest about respondent concerns?
        - Are there any contradictions between sentiment and emotions?
        """)

    # Clear chat button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔄 Reset Chat", key="reset_viz_chat"):
            st.session_state.viz_chat_history = []
            st.rerun()


# --------------------------
# 🤖 COMPARATIVE ANALYSIS INTERPRETER (NEW)
# --------------------------
def create_comparative_interpreter(df_filtered, comparison_config):
    """Chatbot specifically for interpreting comparative analysis"""

    st.markdown("---")
    st.markdown('<h2 class="section-header">🤖 Comparative Analysis Interpreter</h2>', unsafe_allow_html=True)

    # Extract comparison metrics
    col1, col2 = comparison_config.get('columns', ['AI_Challenges', 'AI_Opportunities'])
    demographic = comparison_config.get('demographic', 'Region')
    metric = comparison_config.get('metric', 'Sentiment')

    # Create comparative context
    comparative_context = f"""
    COMPARATIVE ANALYSIS CONTEXT:

    COMPARISON TYPE: {metric.upper()} COMPARISON
    COMPARING: {col1.replace('_', ' ')} vs {col2.replace('_', ' ')}
    BY: {demographic}

    DATA SUMMARY:
    - Total responses: {len(df_filtered)}
    - Demographic groups: {df_filtered[demographic].nunique()}
    - Available for comparison: {df_filtered[demographic].dropna().unique().tolist()}
    """

    # Initialize comparative chat history
    if "comp_chat_history" not in st.session_state:
        st.session_state.comp_chat_history = []

    # Display chat history
    for message in st.session_state.comp_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    comp_prompt = st.chat_input("Ask about the comparative analysis...")

    if comp_prompt:
        # Add user message to history
        st.session_state.comp_chat_history.append({"role": "user", "content": comp_prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(comp_prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            system_prompt = f"""
            You are a comparative analysis interpreter specializing in survey data comparisons.

            {comparative_context}

            GUIDELINES:
            1. Explain differences and similarities between the compared items
            2. Highlight which demographic groups show the most variation
            3. Suggest reasons for observed patterns
            4. Connect comparative findings to practical implications
            5. Point out any unexpected or noteworthy comparisons

            User's question: {comp_prompt}

            Provide a helpful comparative interpretation:
            """

            try:
                response = model.generate_content(system_prompt)
                full_response = response.text
                st.markdown(full_response)
                st.session_state.comp_chat_history.append({"role": "assistant", "content": full_response})

            except Exception as e:
                error_msg = f"Error generating interpretation: {str(e)}"
                st.error(error_msg)
                st.session_state.comp_chat_history.append({"role": "assistant", "content": error_msg})

    # Add example questions
    with st.expander("💡 Comparative analysis questions"):
        st.markdown("""
        **Comparison Questions:**
        - Which group shows the most positive sentiment about AI opportunities?
        - How do challenges differ by institution type?
        - What are the key differences between regions?
        - Which demographic has the most concerns about AI?
        - How do emotions vary across different positions?
        """)

    # Clear chat button
    if st.button("🔄 Reset Comparative Chat", key="reset_comp_chat"):
        st.session_state.comp_chat_history = []
        st.rerun()


# --------------------------
# 🤖 CHATBOT INTEGRATION (DYNAMIC DATA QUERYING)
# --------------------------
# Configure the Generative AI API with Streamlit secrets
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except KeyError:
    st.error("Please add your Google API key to the .streamlit/secrets.toml file.")
    st.stop()

# Set up the model
model = genai.GenerativeModel('models/gemini-1.5-flash-latest')


# NEW FUNCTION: Dynamic data querying for specific questions
def query_survey_data(df, question):
    """Dynamically query the survey data based on the user's question"""
    if df is None or len(df) == 0:
        return "No data available to query."

    # Convert question to lowercase for easier matching
    question_lower = question.lower()

    # Initialize result dictionary
    result = {
        'question_type': 'general',
        'data_found': False,
        'summary': '',
        'specific_data': {},
        'sample_responses': []
    }

    try:
        # Check for country-specific queries
        countries = df['Country'].dropna().unique()
        mentioned_country = None
        for country in countries:
            if country.lower() in question_lower:
                mentioned_country = country
                break

        # Check for region-specific queries
        regions = df['Region'].dropna().unique()
        mentioned_region = None
        for region in regions:
            if region.lower() in question_lower:
                mentioned_region = region
                break

        # Check for institution type queries
        institution_types = df['Institution_Type'].dropna().unique()
        mentioned_institution = None
        for inst_type in institution_types:
            if inst_type.lower() in question_lower:
                mentioned_institution = inst_type
                break

        # Filter data based on mentioned criteria
        filtered_df = df.copy()
        if mentioned_country:
            filtered_df = filtered_df[filtered_df['Country'] == mentioned_country]
            result['specific_data']['country'] = mentioned_country
        if mentioned_region:
            filtered_df = filtered_df[filtered_df['Region'] == mentioned_region]
            result['specific_data']['region'] = mentioned_region
        if mentioned_institution:
            filtered_df = filtered_df[filtered_df['Institution_Type'] == mentioned_institution]
            result['specific_data']['institution_type'] = mentioned_institution

        # Check what type of question this is
        if any(keyword in question_lower for keyword in ['challenge', 'problem', 'issue', 'difficulty']):
            result['question_type'] = 'challenges'
            col = 'AI_Challenges'
        elif any(keyword in question_lower for keyword in ['opportunity', 'benefit', 'advantage', 'positive']):
            result['question_type'] = 'opportunities'
            col = 'AI_Opportunities'
        elif any(keyword in question_lower for keyword in ['enhance', 'improve', 'positive impact', 'benefit']):
            result['question_type'] = 'enhancements'
            col = 'Enhanced_Learning'
        elif any(keyword in question_lower for keyword in ['negative', 'worse', 'problem', 'issue']):
            result['question_type'] = 'negative_impacts'
            col = 'Negative_Impact'
        else:
            result['question_type'] = 'objectives'
            col = 'Objectives'

        # Get relevant data
        relevant_data = filtered_df[col].dropna()
        if len(relevant_data) > 0:
            result['data_found'] = True
            result['response_count'] = len(relevant_data)

            # Get sentiment analysis
            sentiment_col = col + '_sentiment'
            if sentiment_col in filtered_df.columns:
                avg_sentiment = filtered_df[sentiment_col].mean()
                result['specific_data']['avg_sentiment'] = round(avg_sentiment, 3)

            # Get emotion analysis
            emotion_col = col + '_emotion'
            if emotion_col in filtered_df.columns:
                emotion_counts = filtered_df[emotion_col].value_counts().to_dict()
                result['specific_data']['emotions'] = emotion_counts

            # Get sample responses (first 3 non-empty ones)
            sample_responses = relevant_data.head(3).tolist()
            result['sample_responses'] = sample_responses

            # Extract common themes from responses
            if len(relevant_data) > 0:
                themes = extract_meaningful_topics(relevant_data.tolist(), num_topics=3, num_keywords=5)
                result['specific_data']['themes'] = themes

            # Create summary
            location_info = ""
            if mentioned_country:
                location_info = f" in {mentioned_country}"
            elif mentioned_region:
                location_info = f" in the {mentioned_region} region"

            result[
                'summary'] = f"Found {len(relevant_data)} responses about {col.replace('_', ' ').lower()}{location_info}."

    except Exception as e:
        result['summary'] = f"Error querying data: {str(e)}"

    return result


# NEW FUNCTION: Enhanced chatbot response with dynamic data querying
def get_dynamic_chatbot_response(prompt, df, chat_session):
    """Get chatbot response with dynamic data querying capabilities"""

    if df is None:
        return "Please upload a survey data file first to ask questions about the data."

    # First, query the data dynamically based on the question
    query_result = query_survey_data(df, prompt)

    # Create enhanced context based on the query results
    if query_result['data_found']:
        data_context = f"""
        DYNAMIC DATA QUERY RESULTS:

        QUESTION ANALYSIS:
        - Question type: {query_result['question_type']}
        - Responses found: {query_result['response_count']}
        - Summary: {query_result['summary']}

        SPECIFIC DATA:
        {json.dumps(query_result['specific_data'], indent=2)}

        SAMPLE RESPONSES:
        {json.dumps(query_result['sample_responses'], indent=2)}

        FULL DATASET CONTEXT:
        - Total responses: {len(df)}
        - Countries: {df['Country'].nunique()}
        - Regions: {df['Region'].nunique()}
        - Institution types: {df['Institution_Type'].nunique()}
        """
    else:
        data_context = f"""
        DATA CONTEXT:
        - No specific data found matching your query criteria
        - Available in dataset: {len(df)} total responses
        - Countries: {list(df['Country'].dropna().unique())[:5]}... 
        - Regions: {list(df['Region'].dropna().unique())}
        - Try asking about: AI challenges, opportunities, enhanced learning, negative impacts, or objectives

        GENERAL DATASET INFO:
        {json.dumps({
            'total_responses': len(df),
            'available_countries': list(df['Country'].dropna().unique()),
            'available_regions': list(df['Region'].dropna().unique()),
            'available_institution_types': list(df['Institution_Type'].dropna().unique())
        }, indent=2)}
        """

    # Enhanced system prompt
    system_prompt = f"""
    You are an AI assistant analyzing survey data about AI and digital learning in higher education.

    {data_context}

    GUIDELINES FOR RESPONSE:
    1. BASE YOUR ANSWER STRICTLY ON THE DATA PROVIDED ABOVE
    2. If specific data was found, reference the exact numbers and samples
    3. If no specific data was found, suggest alternative questions based on available data
    4. Mention sentiment and emotion trends when relevant
    5. Quote specific sample responses when they illustrate important points
    6. Be concise but informative
    7. If the user asks about a country/region/institution not in the data, politely inform them

    Current user question: {prompt}

    Provide a helpful response based exclusively on the data above:
    """

    try:
        response = chat_session.send_message(system_prompt, stream=True)
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"


# --------------------------
# 🎨 STREAMLIT APP STARTS HERE
# --------------------------
st.markdown('<h1 class="main-header">🎓 Interactive Text Analysis: Generative AI in Higher Education</h1>',
            unsafe_allow_html=True)

# Add the summary paragraph here
st.markdown("""
<div class="highlight">
<p>
This application is designed to simplify understanding survey data about AI and digital learning. It automatically reads key details from an Excel file, such as the respondent's <b>job title</b>, <b>type of school</b>, and <b>location</b>. Its core function is to analyze open-ended text answers about topics like <b>AI challenges</b> and <b>opportunities</b>, as well as the <b>positive</b> and <b>negative impacts</b> of technology on education. Instead of just showing raw data, it presents these insights through visual reports that are easy for anyone to understand. For instance, a <b>word cloud</b> highlights the most talked-about concepts by making the words bigger, while <b>sentiment</b> and <b>emotion charts</b> provide a quick look at the general mood and feelings behind the responses. To help you spot trends, the app also includes <b>comparative charts</b> that let you see side-by-side differences, for example, comparing the overall mood towards "AI challenges" versus "AI opportunities." All of these tools work together to help you quickly uncover the main ideas and feelings expressed in the survey, without needing a data science background.
</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state for data
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_filtered' not in st.session_state:
    st.session_state.df_filtered = None
if 'text_cols' not in st.session_state:
    st.session_state.text_cols = None

df = None
text_cols = None
uploaded_file = None

# --- SIDEBAR NAVIGATION AND FILE UPLOADER ---
with st.sidebar.expander("📁 Upload Data", expanded=True):
    uploaded_file = st.file_uploader(
        "Upload your Excel survey file",
        type=["xlsx"],
        help="Please upload the survey data in Excel format with the required columns"
    )

    if uploaded_file:
        try:
            df, text_cols = load_data(uploaded_file)
            if df is not None and validate_dataframe(df):
                st.success("✅ File successfully loaded and validated!")
                # Store data in session state for chatbot access
                st.session_state.df = df
                st.session_state.df_filtered = df.copy()
                st.session_state.text_cols = text_cols
            else:
                st.error("❌ File validation failed. Please check the format.")
                df = None
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            df = None
    else:
        st.info("""
        **Expected Data Format:**
        - Region
        - Country
        - Position of the respondent...
        - Which category best describes...
        - Which are the key objectives...
        - Enhanced learning experience...
        - Negative learning impact...
        - AI challenges...
        - AI opportunities...
        """)

# Use session state data for the rest of the app
if st.session_state.df is not None:
    df = st.session_state.df
    df_filtered = st.session_state.df_filtered
    text_cols = st.session_state.text_cols

    # --- FILTERING SECTION ---
    with st.sidebar.expander("🔍 Filters", expanded=True):
        st.markdown("### Filter Responses")
        region_filter = st.multiselect(
            "Region", options=df["Region"].dropna().unique(), help="Filter by geographic region"
        )
        country_filter = st.multiselect(
            "Country", options=df["Country"].dropna().unique(), help="Filter by country"
        )
        institution_filter = st.multiselect(
            "Institution Type", options=df["Institution_Type"].dropna().unique(), help="Filter by type of institution"
        )
        position_filter = st.multiselect(
            "Position", options=df["Position"].dropna().unique(), help="Filter by respondent's position"
        )

    # Apply filters
    df_filtered = df.copy()
    if region_filter:
        df_filtered = df_filtered[df_filtered["Region"].isin(region_filter)]
    if country_filter:
        df_filtered = df_filtered[df_filtered["Country"].isin(country_filter)]
    if position_filter:
        df_filtered = df_filtered[df_filtered["Position"].isin(position_filter)]
    if institution_filter:
        df_filtered = df_filtered[df_filtered["Institution_Type"].isin(institution_filter)]

    # Update session state with filtered data
    st.session_state.df_filtered = df_filtered

    st.sidebar.markdown(f"**Filtered Responses:** {len(df_filtered)} of {len(df)}")

    # --- TABBED INTERFACE FOR ANALYSIS SECTIONS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🔍 Question Analysis",
        "📈 Comparative Analysis",
        "📋 Data Explorer",
        "💬 AI Chatbot"
    ])

    # --- TAB 1: OVERVIEW ---
    with tab1:
        st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        with metrics_col1:
            st.metric("Total Responses", len(df), help="Total number of survey responses")
        with metrics_col2:
            st.metric("Regions", df["Region"].nunique(), help="Number of unique regions represented")
        with metrics_col3:
            st.metric("Countries", df["Country"].nunique(), help="Number of unique countries represented")
        with metrics_col4:
            st.metric("Institution Types", df["Institution_Type"].nunique(),
                      help="Number of different institution types")

        st.markdown("---")
        st.markdown('<h3 class="section-header">💡 Quick Insights</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            most_common_region = df["Region"].mode()[0] if not df["Region"].mode().empty else "N/A"
            st.info(f"**Most common region:** {most_common_region}")
            avg_sentiment = df["Objectives_sentiment"].mean()
            sentiment_label = "Positive" if avg_sentiment > 0.3 else ("Negative" if avg_sentiment < -0.3 else "Neutral")
            st.info(f"**Overall sentiment:** {sentiment_label} ({avg_sentiment:.2f})")
        with col2:
            most_common_inst = df["Institution_Type"].mode()[0] if not df["Institution_Type"].mode().empty else "N/A"
            st.info(f"**Most common institution type:** {most_common_inst}")
            complete_responses = df.dropna().shape[0]
            completeness = (complete_responses / len(df)) * 100
            st.info(f"**Complete responses:** {completeness:.1f}%")

        # Display sample data
        st.markdown("---")
        st.markdown('<h3 class="section-header">📋 Sample Data</h3>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

    # --- TAB 2: QUESTION ANALYSIS ---
    with tab2:
        st.markdown('<h2 class="section-header">🔍 Question Analysis</h2>', unsafe_allow_html=True)
        question_map = {
            "Key Objectives (Digital Innovation)": "Objectives",
            "Enhanced Learning Experience": "Enhanced_Learning",
            "Negative Learning Impact": "Negative_Impact",
            "AI Challenges": "AI_Challenges",
            "AI Opportunities": "AI_Opportunities"
        }
        selected_label = st.selectbox(
            "Choose a question to analyze", list(question_map.keys()),
            help="Select a survey question to explore in detail"
        )
        selected_col = question_map[selected_label]
        selected_col_clean = selected_col + "_clean"
        selected_col_sentiment = selected_col + "_sentiment"
        selected_col_emotion = selected_col + "_emotion"
        responses = df_filtered[selected_col_clean].tolist()
        sentiments = df_filtered[selected_col_sentiment].tolist()
        emotions = df_filtered[selected_col_emotion].tolist()

        # Call the corrected word cloud function
        st.markdown("---")
        st.markdown('<h2 class="section-header">☁️ Word Cloud</h2>', unsafe_allow_html=True)
        interactive_word_cloud(responses, f"Most Frequent Words: {selected_label}")

        st.markdown("---")
        st.markdown('<h2 class="section-header">😊 Sentiment Analysis</h2>', unsafe_allow_html=True)
        sentiment_chart = create_enhanced_sentiment_visualization(sentiments,
                                                                  f"Sentiment Distribution: {selected_label}")
        if sentiment_chart:
            st.plotly_chart(sentiment_chart, use_container_width=True)
            st.metric("Average Sentiment Score", f"{np.mean(sentiments):.3f}",
                      help="Average sentiment score across all responses (-1 to +1 scale)")
        else:
            st.warning("No sentiment data available.")

        st.markdown("---")
        st.markdown('<h2 class="section-header">😢 Emotion Analysis</h2>', unsafe_allow_html=True)
        emotion_chart = create_enhanced_emotion_visualization(emotions, f"Emotion Distribution: {selected_label}")
        if emotion_chart:
            st.plotly_chart(emotion_chart, use_container_width=True)
        else:
            st.warning("No emotion data available.")

        st.markdown("---")
        st.markdown('<h2 class="section-header">🧠 Thematic Analysis</h2>', unsafe_allow_html=True)
        themes = extract_meaningful_topics(responses, num_topics=5, num_keywords=5)
        if themes:
            for theme, words in themes.items():
                word_list = ", ".join([f"{word} ({count})" for word, count in words[:5]])
                st.markdown(f"**{theme}**: {word_list}")
            topic_data = []
            for theme, words in themes.items():
                for word, count in words:
                    topic_data.append({"Theme": theme, "Word": word, "Frequency": count})
            topic_df = pd.DataFrame(topic_data)
            fig_topics = px.bar(topic_df, x="Frequency", y="Word", color="Theme", orientation='h',
                                title="Theme Keywords and Frequencies",
                                color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_topics, use_container_width=True)
        else:
            st.warning("Not enough text data to perform thematic analysis.")

        generate_insights_summary(df_filtered, selected_col)

        # ADD THE NEW VISUALIZATION INTERPRETER CHATBOT HERE
        create_visualization_interpreter(df_filtered, selected_col, sentiment_chart, emotion_chart, themes, "question")

        add_visualization_help()

    # --- TAB 3: COMPARATIVE ANALYSIS ---
    with tab3:
        st.markdown('<h2 class="section-header">📈 Interactive Comparative Analysis</h2>', unsafe_allow_html=True)
        st.info("""
        **Comparative Analysis** allows you to compare responses across different demographic groups
        or between different questions. Use the options below to explore patterns and differences.
        """)

        col_demographic, col_metric = st.columns(2)
        with col_demographic:
            demographic_options = ["Region", "Country", "Position", "Institution_Type"]
            selected_demographic = st.selectbox("Select a Demographic to Compare", demographic_options)
        with col_metric:
            metric_options = ["Sentiment", "Emotions", "Top Words"]
            selected_metric = st.selectbox("Select a Metric to Analyze", metric_options)

        if selected_metric in ["Sentiment", "Top Words"]:
            st.markdown("---")
            st.subheader("Select a Topic to Compare")
            topic_pairs = {
                "AI Challenges vs Opportunities": ("AI_Challenges", "AI_Opportunities"),
                "Enhanced Learning vs Negative Impact": ("Enhanced_Learning", "Negative_Impact")
            }
            selected_topic_label = st.selectbox("Choose a topic pair", list(topic_pairs.keys()))
            col1, col2 = topic_pairs[selected_topic_label]

            if selected_metric == "Sentiment":
                # Create sentiment labels for the selected column
                sentiment_col = col1 + "_sentiment"
                df_sent = df_filtered.copy()
                df_sent['Sentiment_Label'] = df_sent[sentiment_col].apply(
                    lambda x: 'Positive' if x > 0.3 else ('Negative' if x < -0.3 else 'Neutral'))

                st.markdown("### Overall Sentiment Comparison")
                fig_box = create_comparative_box_plot(df_filtered, col1, col2,
                                                      f"Sentiment of '{col1.replace('_', ' ')}' vs '{col2.replace('_', ' ')}'")
                st.plotly_chart(fig_box, use_container_width=True)

                st.markdown(f"### Sentiment Breakdown by {selected_demographic}")
                fig_sent_sunburst = create_comparative_sunburst(df_sent, selected_demographic, 'Sentiment_Label')
                st.plotly_chart(fig_sent_sunburst, use_container_width=True)

            elif selected_metric == "Top Words":
                st.markdown(f"### Top Words Comparison")
                create_top_words_comparison(df_filtered, selected_demographic, col1)
                create_top_words_comparison(df_filtered, selected_demographic, col2)

        elif selected_metric == "Emotions":
            st.markdown(f"### Emotion Breakdown by {selected_demographic}")
            # Use the first available emotion column for demonstration
            emotion_col = "AI_Challenges_emotion" if "AI_Challenges_emotion" in df_filtered.columns else "Objectives_emotion"
            fig_emotion_sunburst = create_comparative_sunburst(df_filtered, selected_demographic, emotion_col)
            st.plotly_chart(fig_emotion_sunburst, use_container_width=True)

        # ADD THE COMPARATIVE INTERPRETER CHATBOT HERE
        comparison_config = {
            'columns': [col1, col2] if 'col1' in locals() and 'col2' in locals() else ['AI_Challenges',
                                                                                       'AI_Opportunities'],
            'demographic': selected_demographic,
            'metric': selected_metric
        }
        create_comparative_interpreter(df_filtered, comparison_config)

    # --- TAB 4: DATA EXPLORER ---
    with tab4:
        st.markdown('<h2 class="section-header">📋 Data Explorer</h2>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=400, hide_index=True)

        st.markdown("---")
        st.markdown("### 📝 Data Summary")
        st.json({
            "Total Records": len(df),
            "Columns": list(df.columns),
            "Data Types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        })

        st.markdown("---")
        st.markdown("### 🔍 Column Explorer")
        selected_column = st.selectbox("Select a column to explore", df.columns)
        if selected_column:
            st.write(f"**Unique values in {selected_column}:**")
            st.write(df[selected_column].value_counts())

    # --- TAB 5: AI Chatbot (DYNAMIC DATA QUERYING) ---
    with tab5:
        st.markdown('<h2 class="section-header">💬 Ask the AI Chatbot about the Data</h2>', unsafe_allow_html=True)
        st.info(
            "Ask me specific questions like: 'What were the main challenges identified with AI in Sweden?' or 'How do universities feel about AI opportunities?'")

        # Initialize chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Initialize Gemini chat session in session state
        if "gemini_chat" not in st.session_state:
            st.session_state.gemini_chat = model.start_chat(history=[])

        # Display data status
        if df is not None:
            st.success(f"✅ Chatbot has access to {len(df)} survey responses from {df['Country'].nunique()} countries")
            st.info(
                f"📊 Available countries: {', '.join(list(df['Country'].dropna().unique())[:5])}{'...' if len(df['Country'].dropna().unique()) > 5 else ''}")
        else:
            st.warning("⚠️ Please upload a data file first to enable chatbot functionality")

        # Display chat messages from history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("What would you like to know about the survey data?"):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response from the LLM
            with st.chat_message("assistant"):
                try:
                    response_placeholder = st.empty()
                    full_response = ""

                    # Use dynamic chatbot function with data querying
                    response = get_dynamic_chatbot_response(prompt, df, st.session_state.gemini_chat)

                    if hasattr(response, '__iter__'):
                        # Streaming response
                        for chunk in response:
                            full_response += chunk.text
                            response_placeholder.markdown(full_response + "▌")
                            time.sleep(0.01)
                    else:
                        # Direct response (error message)
                        full_response = response

                    # Display final response
                    response_placeholder.markdown(full_response)

                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

        # Add a clear chat button to reset the conversation
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.session_state.gemini_chat = model.start_chat(history=[])
                st.rerun()

        with col2:
            if st.button("🔍 Test Data Query"):
                if df is not None:
                    test_question = "What are the main AI challenges in Sweden?"
                    query_result = query_survey_data(df, test_question)
                    st.json(query_result)

    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="📥 Download Filtered Data as CSV",
        data=df_filtered.to_csv(index=False).encode('utf-8'),
        file_name="filtered_survey_data.csv",
        mime="text/csv"
    )

# --------------------------
# 💡 FOOTER
# --------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + VADER + Ekman Emotion Model + Gemini AI | For research purposes only")
