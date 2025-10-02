import subprocess
import sys
import os
import hashlib
import re
from collections import Counter
import json


# --------------------------
# 🎯 GROQ AI SETUP - FREE AND FAST AI API
# --------------------------
def install_package(package):
    """Safely install Python packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False


def setup_groq_ai():
    """Safe Groq AI setup that won't crash the app"""
    try:
        import openai
        GROQ_AI_AVAILABLE = True
        print("✅ OpenAI library imported successfully for Groq")
    except ImportError:
        print("📦 OpenAI library not found. Installing...")
        try:
            if install_package("openai"):
                import openai
                GROQ_AI_AVAILABLE = True
                print("✅ OpenAI library installed successfully for Groq")
            else:
                GROQ_AI_AVAILABLE = False
                openai = None
                print("❌ Failed to install OpenAI library")
        except Exception as e:
            GROQ_AI_AVAILABLE = False
            openai = None
            print(f"❌ Installation error: {e}")

    return GROQ_AI_AVAILABLE, openai if 'openai' in locals() else None


# Initialize Groq AI at module level
GROQ_AI_AVAILABLE, openai = setup_groq_ai()

# Install other required packages
required_packages = ["plotly", "nltk", "wordcloud", "matplotlib", "pillow", "pandas", "numpy", "streamlit"]
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        install_package(package)

# Now import other packages
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# --------------------------
# ⚙️ STREAMLIT PAGE CONFIG MUST BE THE FIRST COMMAND
# --------------------------
st.set_page_config(
    page_title="AI & Digital Learning Insights",
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

# --------------------------
# 🛠️ NLTK DATA DOWNLOADS
# --------------------------
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

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
    .quick-guide {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    @media (max-width: 768px) {
        .main-header { font-size: 2rem !important; }
        .section-header { font-size: 1.4rem !important; }
    }
</style>
""", unsafe_allow_html=True)


# --------------------------
# 🔧 UPDATED GROQ API FUNCTIONS
# --------------------------
def configure_groq_api():
    """Safe API configuration that won't crash"""
    if not GROQ_AI_AVAILABLE or openai is None:
        return False, "OpenAI library not available for Groq"

    api_key = None

    # 1. Check session state first
    if 'groq_api_key' in st.session_state and st.session_state.groq_api_key:
        api_key = st.session_state.groq_api_key

    # 2. Check Streamlit secrets
    if not api_key:
        try:
            if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
                api_key = st.secrets['GROQ_API_KEY']
        except:
            pass

    # 3. Check environment variable
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if api_key and api_key.strip():
        try:
            # Configure OpenAI client for Groq
            client = openai.OpenAI(
                api_key=api_key.strip(),
                base_url="https://api.groq.com/openai/v1"
            )
            # Store client in session state for reuse
            st.session_state.groq_client = client
            return True, "✅ Groq API Active"
        except Exception as e:
            return False, f"❌ API Error: {str(e)[:50]}..."

    return False, "🔑 No API key configured"


def safe_groq_response(prompt, context=""):
    """Generate AI response with Groq API and comprehensive error handling"""
    if not GROQ_AI_AVAILABLE:
        return "🤖 AI features currently unavailable. Please check OpenAI library installation."

    # Check if API is configured
    api_configured, status = configure_groq_api()
    if not api_configured:
        return f"🔧 {status}"

    try:
        # Get the configured client
        client = st.session_state.get('groq_client')
        if not client:
            return "❌ Groq client not configured"

        # Try different Groq models in order of preference
        model_names = [
            'llama-3.1-70b-versatile',  # Latest Llama model
            'llama-3.1-8b-instant',  # Faster Llama model
            'mixtral-8x7b-32768',  # Mixtral model
            'gemma2-9b-it',  # Gemma model
        ]

        response_text = None
        last_error = None

        for model_name in model_names:
            try:
                full_prompt = f"""
                You are a data analysis assistant for survey data. Please provide helpful, data-driven insights.

                Data Context: {context}

                User Question: {prompt}

                Please provide a concise, informative response based on the available data.
                """

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system",
                         "content": "You are a helpful data analysis assistant specializing in survey data insights."},
                        {"role": "user", "content": full_prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )

                if response.choices and response.choices[0].message.content:
                    response_text = response.choices[0].message.content
                    break  # Success, exit loop

            except Exception as e:
                last_error = e
                continue  # Try next model

        if response_text:
            return response_text
        else:
            return f"❌ All models failed. Last error: {str(last_error)[:100]}..."

    except Exception as e:
        error_msg = str(e)

        # Handle specific error types with user-friendly messages
        if "invalid_api_key" in error_msg.lower() or "unauthorized" in error_msg.lower():
            return "❌ Invalid API key. Please check your Groq API key in the sidebar."
        elif "quota" in error_msg.lower() or "rate_limit" in error_msg.lower():
            return """📊 **Rate limit reached**. Groq has generous free limits, but you may have exceeded them.

**Solutions:**
- Wait a few minutes and try again
- Get a paid Groq account for higher limits
- The free tier resets daily"""
        elif "content_filter" in error_msg.lower():
            return "🛡️ Content safety filters triggered. Please rephrase your question."
        elif "404" in error_msg or "not found" in error_msg.lower():
            return "🔄 Model not found. Trying different Groq models automatically..."
        elif "503" in error_msg or "500" in error_msg:
            return "🌐 Groq service temporarily unavailable. Please try again in a few moments."
        else:
            return f"⚠️ AI service temporarily unavailable. Error: {error_msg[:80]}..."


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


# --- ENHANCED INTERACTIVE WORD FREQUENCY SCATTER PLOT ---
def interactive_word_cloud(text_data, title):
    """Create an enhanced interactive word frequency visualization"""

    if not text_data:
        st.warning("No text data to analyze.")
        return

    full_text = " ".join([t for t in text_data if t])
    if not full_text:
        st.warning("No valid text to analyze.")
        return

    # ----------------  ENHANCED CONTROLS  ----------------
    st.markdown("### 🎛️ Analysis Controls")
    col1, col2, col3 = st.columns(3)

    with col1:
        max_words = st.slider("Words to show", 10, 100, 50, key=f"{title}_words")
    with col2:
        min_word_length = st.slider("Min word length", 3, 8, 4, key=f"{title}_length")
    with col3:
        chart_type = st.selectbox("Chart type", ["Scatter Plot", "Bar Chart", "Treemap"], key=f"{title}_chart")

    # ----------------  PROCESS TEXT  ----------------
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'university', 'education', 'learning', 'student', 'technology',
                        'digital', 'institution', 'experience', 'use', 'using'}
    all_stopwords = stop_words.union(custom_stopwords)

    words = re.findall(rf'\b[a-z]{{{min_word_length},}}\b', full_text.lower())
    filtered_words = [word for word in words if word not in all_stopwords]
    word_freq = Counter(filtered_words)
    top_words = word_freq.most_common(max_words)

    if not top_words:
        st.warning("No meaningful words found after filtering.")
        return

    # ----------------  CREATE VISUALIZATION  ----------------
    words_list, counts_list = zip(*top_words)
    df_words = pd.DataFrame({
        'word': words_list,
        'frequency': counts_list,
        'word_length': [len(word) for word in words_list],
        'rank': range(1, len(words_list) + 1)
    })

    with st.spinner('Generating visualization...'):
        if chart_type == "Scatter Plot":
            fig = px.scatter(df_words, x='rank', y='frequency', size='frequency',
                             hover_name='word', title=f"📈 Word Frequency: {title}",
                             size_max=30, color='frequency', color_continuous_scale='viridis')
            fig.update_layout(xaxis_title="Word Rank", yaxis_title="Frequency", height=500)

        elif chart_type == "Bar Chart":
            fig = px.bar(df_words.head(30), x='frequency', y='word', orientation='h',
                         title=f"📊 Top Words: {title}", color='frequency',
                         color_continuous_scale='plasma')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)

        else:  # Treemap
            fig = px.treemap(df_words, path=['word'], values='frequency',
                             title=f"🌳 Word Treemap: {title}", color='frequency',
                             color_continuous_scale='rainbow')
            fig.update_layout(height=500)

        st.plotly_chart(fig, use_container_width=True)

    # ----------------  ENHANCED ANALYSIS SECTION  ----------------
    with st.expander("📈 Detailed Word Analysis", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Unique Words", len(word_freq))
        with col2:
            st.metric("Most Frequent", f"'{words_list[0]}'")
        with col3:
            st.metric("Max Frequency", counts_list[0])
        with col4:
            diversity = len(word_freq) / len([w for w in full_text.split() if len(w) > 3]) if full_text else 0
            st.metric("Diversity", f"{diversity:.1%}" if diversity > 0 else "N/A")

        # Word frequency table
        st.subheader("📋 Word Frequency Table")
        display_df = df_words.head(20).copy()
        st.dataframe(display_df[['rank', 'word', 'frequency', 'word_length']],
                     use_container_width=True, height=300)


# --------------------------
# 🎯 LOAD AND PREPARE DATA
# --------------------------
@st.cache_data
def load_data(file):
    try:
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
                st.error(f"Missing column: {col}")
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

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None


# --------------------------
# 🎨 ENHANCED VISUALIZATION FUNCTIONS
# --------------------------
def validate_dataframe(df):
    if df is None or len(df) == 0:
        return False
    required_cols = ["Region", "Country", "Position", "Institution_Type",
                     "Objectives", "Enhanced_Learning", "Negative_Impact",
                     "AI_Challenges", "AI_Opportunities"]
    return all(col in df.columns for col in required_cols)


def create_enhanced_sentiment_visualization(sentiments, title):
    if not sentiments or not any(isinstance(s, (int, float)) for s in sentiments):
        return None, None

    # Create sentiment gauge
    avg_sentiment = np.mean(sentiments)
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_sentiment,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Average Sentiment"},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.3], 'color': "lightcoral"},
                {'range': [-0.3, 0.3], 'color': "lightyellow"},
                {'range': [0.3, 1], 'color': "lightgreen"}]
        }
    ))
    gauge_fig.update_layout(height=300)

    # Create histogram
    hist_fig = px.histogram(x=sentiments, nbins=20, title=title,
                            labels={"x": "Sentiment Score (-1 = Negative, +1 = Positive)"})
    hist_fig.add_vline(x=0, line_dash="dash", line_color="gray")
    hist_fig.add_vline(x=0.5, line_dash="dot", line_color="green")
    hist_fig.add_vline(x=-0.5, line_dash="dot", line_color="red")

    return gauge_fig, hist_fig


def create_enhanced_emotion_visualization(emotions, title):
    if not emotions:
        return None
    emotion_counts = pd.Series(emotions).value_counts()
    emotion_colors = {
        'joy': '#4CAF50', 'surprise': '#FFC107', 'neutral': '#9E9E9E',
        'sadness': '#2196F3', 'fear': '#673AB7', 'anger': '#F44336', 'disgust': '#795548'
    }
    fig = px.pie(values=emotion_counts.values, names=emotion_counts.index, title=title,
                 color=emotion_counts.index, color_discrete_map=emotion_colors,
                 hole=0.3)  # Donut chart
    fig.update_traces(textposition='inside', textinfo='percent+label')
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
    st.subheader("💡 Key Insights Dashboard")

    # Create insight cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if avg_sentiment > 0.3:
            st.success(f"😊 Positive\n{avg_sentiment:.2f}")
        elif avg_sentiment < -0.3:
            st.error(f"😟 Negative\n{avg_sentiment:.2f}")
        else:
            st.info(f"😐 Mixed\n{avg_sentiment:.2f}")

    with col2:
        detailed_responses = len([r for r in df_filtered[selected_col + "_clean"] if len(str(r)) > 10])
        st.info(f"🗣️ Detailed\n{detailed_responses} responses")

    with col3:
        if primary_emotion == "joy":
            st.success(f"🎭 {primary_emotion.title()}")
        elif primary_emotion in ["anger", 'disgust', 'sadness']:
            st.error(f"🎭 {primary_emotion.title()}")
        else:
            st.info(f"🎭 {primary_emotion.title()}")

    with col4:
        unique_words = len(set(' '.join(df_filtered[selected_col + "_clean"].astype(str)).split()))
        st.success(f"📚 Vocabulary\n{unique_words} words")


# --- COMPARATIVE VISUALIZATIONS ---
def create_comparative_box_plot(df, col1, col2, title):
    df_melted = pd.melt(df, value_vars=[col1 + "_sentiment", col2 + "_sentiment"])
    fig = px.box(df_melted, x="variable", y="value", title=title,
                 color="variable", labels={'variable': 'Topic', 'value': 'Sentiment Score'})
    fig.update_layout(xaxis_title="Topic", yaxis_title="Sentiment Score")
    return fig


def create_comparative_sunburst(df, demographic_col, text_col):
    df_grouped = df.groupby([demographic_col, text_col]).size().reset_index(name='count')
    fig = px.sunburst(df_grouped, path=[demographic_col, text_col], values='count',
                      title=f"Breakdown by {demographic_col}")
    fig.update_layout(height=500)
    return fig


# --------------------------
# 🤖 ENHANCED DYNAMIC AI CHAT SYSTEM WITH GROQ - IMPROVED VERSION
# --------------------------
def get_data_context(df_filtered, selected_col=None):
    """Create comprehensive data context for AI responses"""
    if df_filtered is None or len(df_filtered) == 0:
        return {}

    context = {
        'total_responses': len(df_filtered),
        'regions': df_filtered['Region'].dropna().unique().tolist(),
        'countries': df_filtered['Country'].dropna().unique().tolist(),
        'institution_types': df_filtered['Institution_Type'].dropna().unique().tolist(),
        'positions': df_filtered['Position'].dropna().unique().tolist()
    }

    if selected_col:
        # Add question-specific context
        sentiments = df_filtered[selected_col + "_sentiment"].dropna().tolist()
        emotions = df_filtered[selected_col + "_emotion"].dropna().tolist()
        responses = df_filtered[selected_col + "_clean"].dropna().tolist()

        context.update({
            'current_question': selected_col.replace('_', ' ').title(),
            'avg_sentiment': np.mean(sentiments) if sentiments else 0,
            'sentiment_distribution': {
                'positive': len([s for s in sentiments if s > 0.3]),
                'neutral': len([s for s in sentiments if -0.3 <= s <= 0.3]),
                'negative': len([s for s in sentiments if s < -0.3])
            },
            'emotion_counts': pd.Series(emotions).value_counts().to_dict(),
            'sample_responses': responses[:3] if responses else [],
            'total_words': len(' '.join(responses).split()) if responses else 0,
            'response_lengths': [len(str(r)) for r in responses if r]
        })

    return context


def analyze_user_question(question, data_context):
    """Enhanced question analysis for more dynamic responses"""
    question_lower = question.lower()

    analysis = {
        'question_type': 'general',
        'topics': [],
        'demographics': [],
        'needs_comparison': False,
        'needs_sentiment': False,
        'needs_emotion': False,
        'needs_trends': False,
        'needs_recommendations': False,
        'specific_question': question_lower
    }

    # Enhanced question type detection
    sentiment_words = ['sentiment', 'feeling', 'mood', 'positive', 'negative', 'optimistic', 'pessimistic']
    emotion_words = ['emotion', 'feeling', 'joy', 'anger', 'fear', 'sadness', 'disgust', 'surprise', 'emotional']
    comparison_words = ['compare', 'difference', 'versus', 'vs', 'between', 'contrast', 'versus']
    trend_words = ['trend', 'pattern', 'over time', 'evolution', 'change', 'development']
    recommendation_words = ['recommend', 'suggest', 'advice', 'should we', 'what to do', 'action', 'solution']

    if any(word in question_lower for word in sentiment_words):
        analysis['needs_sentiment'] = True
        analysis['question_type'] = 'sentiment'

    if any(word in question_lower for word in emotion_words):
        analysis['needs_emotion'] = True
        analysis['question_type'] = 'emotion'

    if any(word in question_lower for word in comparison_words):
        analysis['needs_comparison'] = True
        analysis['question_type'] = 'comparison'

    if any(word in question_lower for word in trend_words):
        analysis['needs_trends'] = True
        analysis['question_type'] = 'trends'

    if any(word in question_lower for word in recommendation_words):
        analysis['needs_recommendations'] = True
        analysis['question_type'] = 'recommendations'

    # Enhanced demographic detection
    for region in data_context.get('regions', []):
        if region and isinstance(region, str) and region.lower() in question_lower:
            analysis['demographics'].append(('region', region))

    for country in data_context.get('countries', []):
        if country and isinstance(country, str) and country.lower() in question_lower:
            analysis['demographics'].append(('country', country))

    for inst_type in data_context.get('institution_types', []):
        if inst_type and isinstance(inst_type, str) and inst_type.lower() in question_lower:
            analysis['demographics'].append(('institution_type', inst_type))

    # Detect specific topics
    if 'ai' in question_lower or 'artificial intelligence' in question_lower:
        analysis['topics'].append('AI')
    if 'digital' in question_lower or 'technology' in question_lower:
        analysis['topics'].append('Digital Technology')
    if 'learning' in question_lower or 'education' in question_lower:
        analysis['topics'].append('Learning')
    if 'challenge' in question_lower:
        analysis['topics'].append('Challenges')
    if 'opportunity' in question_lower:
        analysis['topics'].append('Opportunities')

    return analysis


def generate_dynamic_response(question, data_context, analysis, themes=None):
    """Enhanced dynamic response generation with more specific insights"""

    response_parts = []

    # Add personalized greeting based on question type
    if analysis['question_type'] == 'sentiment':
        response_parts.append("**📊 Sentiment Analysis Results:**")
    elif analysis['question_type'] == 'emotion':
        response_parts.append("**🎭 Emotional Tone Analysis:**")
    elif analysis['question_type'] == 'comparison':
        response_parts.append("**⚖️ Comparative Insights:**")
    elif analysis['question_type'] == 'recommendations':
        response_parts.append("**💡 Actionable Recommendations:**")
    else:
        response_parts.append("**🔍 Data Analysis Insights:**")

    # Enhanced sentiment analysis with more detail
    if analysis['needs_sentiment'] and 'avg_sentiment' in data_context:
        avg_sentiment = data_context['avg_sentiment']

        # More nuanced sentiment descriptions
        if avg_sentiment > 0.6:
            sentiment_desc = "strongly positive"
            emoji = "😊"
        elif avg_sentiment > 0.3:
            sentiment_desc = "moderately positive"
            emoji = "🙂"
        elif avg_sentiment > 0.1:
            sentiment_desc = "slightly positive"
            emoji = "😐"
        elif avg_sentiment < -0.6:
            sentiment_desc = "strongly negative"
            emoji = "😠"
        elif avg_sentiment < -0.3:
            sentiment_desc = "moderately negative"
            emoji = "😟"
        elif avg_sentiment < -0.1:
            sentiment_desc = "slightly negative"
            emoji = "😕"
        else:
            sentiment_desc = "neutral"
            emoji = "😐"

        response_parts.append(f"{emoji} **Overall Sentiment**: {sentiment_desc.title()} (score: {avg_sentiment:.3f})")

        dist = data_context.get('sentiment_distribution', {})
        if dist:
            total = sum(dist.values())
            if total > 0:
                response_parts.append(
                    f"   • 📈 **{dist.get('positive', 0)}** positive responses ({dist.get('positive', 0) / total * 100:.1f}%)")
                response_parts.append(
                    f"   • 📊 **{dist.get('neutral', 0)}** neutral responses ({dist.get('neutral', 0) / total * 100:.1f}%)")
                response_parts.append(
                    f"   • 📉 **{dist.get('negative', 0)}** negative responses ({dist.get('negative', 0) / total * 100:.1f}%)")

    # Enhanced emotion analysis
    if analysis['needs_emotion'] and 'emotion_counts' in data_context:
        emotion_counts = data_context['emotion_counts']
        if emotion_counts:
            total_emotions = sum(emotion_counts.values())
            emotion_texts = []
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_emotions) * 100
                emotion_texts.append(f"**{emotion.title()}** ({count}, {percentage:.1f}%)")

            if emotion_texts:
                response_parts.append(f"🎭 **Emotional Distribution**: {', '.join(emotion_texts[:3])}")

    # Demographic-specific insights
    if analysis['demographics']:
        for demo_type, demo_value in analysis['demographics']:
            response_parts.append(
                f"🌍 **{demo_value} Focus**: Responses from this group show distinctive patterns in the data.")

    # Thematic insights with more context
    if themes and analysis['question_type'] != 'comparison':
        theme_list = list(themes.keys())[:4]
        if theme_list:
            response_parts.append(f"🧠 **Key Themes Identified**: {', '.join(theme_list)}")

            # Add some specific keywords from top themes
            top_theme = list(themes.keys())[0]
            if themes[top_theme]:
                top_keywords = [word for word, count in themes[top_theme][:3]]
                response_parts.append(f"   • **{top_theme}**: Features keywords like '{', '.join(top_keywords)}'")

    # Response quality insights
    if 'response_lengths' in data_context and data_context['response_lengths']:
        avg_length = np.mean(data_context['response_lengths'])
        if avg_length > 100:
            response_parts.append("💬 **Response Quality**: Detailed and thoughtful responses")
        elif avg_length > 50:
            response_parts.append("💬 **Response Quality**: Moderately detailed feedback")
        else:
            response_parts.append("💬 **Response Quality**: Brief but focused responses")

    # Data context summary
    response_parts.append(
        f"📈 **Analysis Scope**: Based on {data_context['total_responses']} survey responses across {len(data_context.get('countries', []))} countries")

    # Enhanced actionable insights based on question type
    if analysis['needs_recommendations']:
        if 'avg_sentiment' in data_context:
            avg_sentiment = data_context['avg_sentiment']
            if avg_sentiment > 0.3:
                response_parts.append(
                    "💡 **Strategic Recommendation**: Build on this positive momentum and scale successful initiatives.")
            elif avg_sentiment < -0.3:
                response_parts.append(
                    "💡 **Strategic Recommendation**: Address concerns through targeted interventions and improved communication.")
            else:
                response_parts.append(
                    "💡 **Strategic Recommendation**: Focus on clarifying benefits and providing more support resources.")

    # Add specific insights based on question content
    if any(topic in analysis['specific_question'] for topic in ['ai', 'artificial intelligence']):
        response_parts.append(
            "🤖 **AI-Specific Insight**: Consider both ethical implications and practical applications in your strategy.")

    if 'challenge' in analysis['specific_question']:
        response_parts.append(
            "🛠️ **Challenge Focus**: Prioritize addressing the most frequently mentioned obstacles first.")

    if 'opportunity' in analysis['specific_question']:
        response_parts.append("🚀 **Opportunity Focus**: Leverage these insights to create competitive advantages.")

    return "\n\n".join(response_parts)


def create_dynamic_ai_chat(df_filtered, selected_col=None, themes=None, tab_name="main"):
    """Enhanced AI chat with truly dynamic data-driven responses using Groq"""

    st.markdown("---")
    st.markdown('<h3 class="section-header">💬 Dynamic AI Assistant</h3>', unsafe_allow_html=True)

    # Initialize chat history with unique key for each tab
    chat_history_key = f"dynamic_chat_history_{tab_name}"
    if chat_history_key not in st.session_state:
        st.session_state[chat_history_key] = []

    # Get current data context
    data_context = get_data_context(df_filtered, selected_col)

    # Display AI status
    ai_status, status_msg = configure_groq_api()
    status_color = "🟢" if ai_status else "🔴"
    st.caption(f"{status_color} Groq AI Status: {status_msg}")

    # Display chat history
    for message in st.session_state[chat_history_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Enhanced quick question suggestions with context-awareness
    st.markdown("**💡 Try asking:**")
    col1, col2, col3 = st.columns(3)

    # Context-aware quick questions
    base_questions = [
        "What's the overall sentiment?",
        "Which emotions are most common?",
        "How do responses vary by region?",
        "What are the main themes?",
        "Compare challenges vs opportunities",
        "What actions should we take?"
    ]

    # Add context-specific questions if we have a selected column
    if selected_col:
        question_topic = selected_col.replace('_', ' ').title()
        base_questions = [
            f"What's the sentiment about {question_topic.lower()}?",
            f"What emotions are expressed about {question_topic.lower()}?",
            f"What are the main themes in {question_topic.lower()} responses?",
            f"How do different regions view {question_topic.lower()}?",
            f"What recommendations for {question_topic.lower()}?",
            f"Compare positive and negative aspects of {question_topic.lower()}"
        ]

    for i, question in enumerate(base_questions):
        col = [col1, col2, col3][i % 3]
        with col:
            # Create UNIQUE key using tab_name and question index
            unique_key = f"quick_q_{tab_name}_{i}_{hashlib.md5(question.encode()).hexdigest()[:8]}"
            if st.button(question, key=unique_key, use_container_width=True):
                st.session_state[chat_history_key].append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                # Generate response - use Groq AI if available, otherwise enhanced fallback
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing your data..."):
                        if ai_status:
                            context_text = f"Data Context: {json.dumps(data_context, indent=2)}"
                            response = safe_groq_response(question, context_text)
                        else:
                            analysis = analyze_user_question(question, data_context)
                            response = generate_dynamic_response(question, data_context, analysis, themes)

                        st.markdown(response)

                st.session_state[chat_history_key].append({"role": "assistant", "content": response})
                st.rerun()

    # Chat input with unique key
    chat_input_key = f"chat_input_{tab_name}"
    user_question = st.chat_input("Or ask your own question about the data...", key=chat_input_key)

    if user_question:
        # Add user message to history
        st.session_state[chat_history_key].append({"role": "user", "content": user_question})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data with Groq AI..."):
                # Use Groq AI if available, otherwise enhanced fallback
                if ai_status:
                    context_text = f"Data Context: {json.dumps(data_context, indent=2)}"
                    response = safe_groq_response(user_question, context_text)
                else:
                    analysis = analyze_user_question(user_question, data_context)
                    response = generate_dynamic_response(user_question, data_context, analysis, themes)

                st.markdown(response)

        # Add to chat history
        st.session_state[chat_history_key].append({"role": "assistant", "content": response})
        st.rerun()

    # Chat controls with unique key
    clear_key = f"clear_chat_{tab_name}"
    if st.button("🔄 Clear Chat History", key=clear_key, use_container_width=True):
        st.session_state[chat_history_key] = []
        st.rerun()


# --------------------------
# 🎨 MAIN APPLICATION
# --------------------------
st.markdown('<h1 class="main-header">🎓 AI & Digital Learning Insights Dashboard</h1>',
            unsafe_allow_html=True)

# Quick Start Guide
with st.expander("🚀 **Quick Start Guide**", expanded=True):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        **1. 📁 Upload Data** → Use the sidebar to upload your Excel file  
        **2. 📊 Explore Overview** → See dataset statistics and sample data  
        **3. 🔍 Analyze Questions** → Dive deep into each survey question  
        **4. 📈 Compare Groups** → See differences by demographics  
        **5. 💬 Chat with Data** → Ask natural language questions (Groq AI - FREE!)  
        """)

    with col2:
        st.info("""
        **Pro Tips:**
        - Start with Question Analysis
        - Use filters to focus on specific groups
        - Hover over charts for details
        - Ask the AI assistant for insights
        """)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_filtered' not in st.session_state:
    st.session_state.df_filtered = None
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = ""

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 📁 Data Management")

    with st.expander("Upload Data", expanded=True):
        uploaded_file = st.file_uploader("Choose Excel file", type=["xlsx"],
                                         help="Upload survey data in Excel format")

        if uploaded_file:
            with st.spinner('Loading and analyzing data...'):
                df, text_cols = load_data(uploaded_file)
                if df is not None and validate_dataframe(df):
                    st.session_state.df = df
                    st.session_state.df_filtered = df.copy()
                    st.success("✅ Data loaded successfully!")

                    # Show quick stats
                    st.info(f"""
                    **Data Loaded:**
                    - {len(df)} total responses
                    - {df['Country'].nunique()} countries
                    - {df['Region'].nunique()} regions
                    - {df['Institution_Type'].nunique()} institution types
                    """)
                else:
                    st.error("❌ Invalid file format")

    # Groq API Configuration
    with st.expander("🔐 Groq AI Configuration (FREE)", expanded=True):
        st.markdown("**Configure Groq AI API**")

        # Show current status
        ai_status, status_msg = configure_groq_api()
        status_color = "🟢" if ai_status else "🔴"
        st.info(f"{status_color} **Status**: {status_msg}")

        # Free tier notice
        st.success("🎉 **GROQ**")

        # API key input
        api_key = st.text_input(
            "Enter Groq API Key:",
            type="password",
            placeholder="gsk_...",
            value=st.session_state.groq_api_key,
            help="Get your FREE API key from https://console.groq.com/keys"
        )

        if api_key != st.session_state.groq_api_key:
            st.session_state.groq_api_key = api_key
            if api_key:
                st.success("✅ API key saved! Testing connection...")
                st.rerun()

    if st.session_state.df is not None:
        with st.expander("🔍 Filters", expanded=True):
            df = st.session_state.df
            region_filter = st.multiselect("Region", options=df["Region"].dropna().unique())
            country_filter = st.multiselect("Country", options=df["Country"].dropna().unique())
            institution_filter = st.multiselect("Institution Type", options=df["Institution_Type"].dropna().unique())

            # Apply filters
            df_filtered = df.copy()
            if region_filter:
                df_filtered = df_filtered[df_filtered["Region"].isin(region_filter)]
            if country_filter:
                df_filtered = df_filtered[df_filtered["Country"].isin(country_filter)]
            if institution_filter:
                df_filtered = df_filtered[df_filtered["Institution_Type"].isin(institution_filter)]

            st.session_state.df_filtered = df_filtered
            st.info(f"📊 Showing {len(df_filtered)} of {len(df)} responses")

# --- MAIN CONTENT ---
if st.session_state.df is not None:
    df = st.session_state.df
    df_filtered = st.session_state.df_filtered

    # Tab interface
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Overview", "🔍 Question Analysis", "📈 Comparisons", "📋 Data Explorer", "💬 AI Assistant"])

    with tab1:
        st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)

        # Key metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Responses", len(df), help="Number of survey responses")
        with m2:
            st.metric("Countries", df["Country"].nunique(), "Unique countries")
        with m3:
            st.metric("Institution Types", df["Institution_Type"].nunique(), "Different types")
        with m4:
            complete = df.dropna().shape[0]
            st.metric("Complete Data", f"{(complete / len(df) * 100):.1f}%", "Response completeness")

        # Sample data
        with st.expander("📋 View Sample Data", expanded=True):
            st.dataframe(df.head(8), use_container_width=True)

    with tab2:
        st.markdown('<h2 class="section-header">🔍 Question Analysis</h2>', unsafe_allow_html=True)

        question_map = {
            "AI Challenges": "AI_Challenges",
            "AI Opportunities": "AI_Opportunities",
            "Enhanced Learning": "Enhanced_Learning",
            "Key Objectives": "Objectives",
            "Negative Impact": "Negative_Impact"
        }

        selected_label = st.selectbox("Choose question to analyze:", list(question_map.keys()))
        selected_col = question_map[selected_label]
        responses = df_filtered[selected_col + "_clean"].tolist()

        # Word analysis
        st.markdown("---")
        interactive_word_cloud(responses, f"Analysis: {selected_label}")

        # Sentiment analysis
        st.markdown("---")
        st.markdown('<h3 class="section-header">😊 Sentiment Analysis</h3>', unsafe_allow_html=True)
        sentiments = df_filtered[selected_col + "_sentiment"].tolist()

        if sentiments:
            gauge_fig, hist_fig = create_enhanced_sentiment_visualization(sentiments, "Sentiment Distribution")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(gauge_fig, use_container_width=True)
            with col2:
                st.plotly_chart(hist_fig, use_container_width=True)

            st.metric("Average Sentiment", f"{np.mean(sentiments):.3f}",
                      delta=f"{np.mean(sentiments):.3f} from neutral" if np.mean(sentiments) != 0 else "Neutral")

        # Emotion analysis
        st.markdown("---")
        st.markdown('<h3 class="section-header">😢 Emotion Analysis</h3>', unsafe_allow_html=True)
        emotions = df_filtered[selected_col + "_emotion"].tolist()
        if emotions:
            emotion_fig = create_enhanced_emotion_visualization(emotions, "Emotion Distribution")
            st.plotly_chart(emotion_fig, use_container_width=True)

        # Thematic analysis
        st.markdown("---")
        st.markdown('<h3 class="section-header">🧠 Thematic Analysis</h3>', unsafe_allow_html=True)
        themes = extract_meaningful_topics(responses)
        if themes:
            for theme, words in themes.items():
                word_list = ", ".join([f"{word} ({count})" for word, count in words[:5]])
                st.info(f"**{theme}**: {word_list}")

        # Insights summary
        generate_insights_summary(df_filtered, selected_col)

        # Dynamic AI chat with UNIQUE tab identifier
        create_dynamic_ai_chat(df_filtered, selected_col, themes, "question_analysis")

    with tab3:
        st.markdown('<h2 class="section-header">📈 Comparative Analysis</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            demographic = st.selectbox("Compare by:", ["Region", "Country", "Institution_Type", "Position"],
                                       key="demo_select")
        with col2:
            comparison = st.selectbox("Compare:",
                                      ["AI Challenges vs Opportunities", "Enhanced Learning vs Negative Impact"],
                                      key="comp_select")

        if comparison == "AI Challenges vs Opportunities":
            col1, col2 = "AI_Challenges", "AI_Opportunities"
        else:
            col1, col2 = "Enhanced_Learning", "Negative_Impact"

        # Comparative visualization
        st.subheader("Sentiment Comparison")
        fig = create_comparative_box_plot(df_filtered, col1, col2, f"Comparison: {col1} vs {col2}")
        st.plotly_chart(fig, use_container_width=True)

        # Sunburst chart
        st.subheader(f"Breakdown by {demographic}")
        sunburst_fig = create_comparative_sunburst(df_filtered, demographic, col1 + "_emotion")
        st.plotly_chart(sunburst_fig, use_container_width=True)

    with tab4:
        st.markdown('<h2 class="section-header">📋 Data Explorer</h2>', unsafe_allow_html=True)

        st.dataframe(df, use_container_width=True, height=400)

        st.markdown("---")
        st.subheader("Data Summary")
        col1, col2 = st.columns(2)

        with col1:
            st.json({
                "Total Records": len(df),
                "Columns": list(df.columns),
                "Data Types": {col: str(dtype) for col, dtype in df.dtypes.items()}
            })

        with col2:
            selected_column = st.selectbox("Explore column:", df.columns, key="col_explorer")
            if selected_column:
                st.write(f"**Unique values in {selected_column}:**")
                value_counts = df[selected_column].value_counts()
                st.dataframe(value_counts, use_container_width=True)

    with tab5:
        st.markdown('<h2 class="section-header">💬 Dynamic AI Analysis Assistant </h2>',
                    unsafe_allow_html=True)

        if df is not None:
            # Show data overview
            st.success(f"✅ Groq AI Assistant has access to {len(df)} survey responses")

            # Data summary cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Responses", len(df))
            with col2:
                st.metric("Countries", df['Country'].nunique())
            with col3:
                st.metric("Regions", df['Region'].nunique())
            with col4:
                st.metric("Institution Types", df['Institution_Type'].nunique())

            # Dynamic AI chat with UNIQUE tab identifier
            create_dynamic_ai_chat(df_filtered, tab_name="main_assistant")

        else:
            st.warning("📁 Please upload a data file first to enable AI analysis")
            st.info("Use the sidebar to upload your Excel survey data file")

else:
    # Welcome screen when no data loaded
    st.markdown("""
    <div class='highlight'>
    <h3>👋 Welcome to the AI & Digital Learning Insights Dashboard!</h3>
    <p>To get started, please upload your survey data using the panel on the left.</p>
    <p><b>Expected data format:</b> Excel file with columns for Region, Country, Position, Institution Type, 
    and survey questions about AI challenges, opportunities, and learning impacts.</p>
    <p><b>🚀 New:</b> This version is powered by Groq AI for ultra-fast, FREE data analysis capabilities!</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption(
    "Built with ❤️ using Streamlit | Groq AI (FREE) | VADER Sentiment Analysis | Ekman Emotions | Interactive Visualizations")

# Performance optimization
st.markdown("""
<style>
    .stSpinner > div { text-align: center; }
    .stButton > button { width: 100%; }
    .stDataFrame { font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)
