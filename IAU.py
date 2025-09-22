import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import os
from collections import Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from textblob import TextBlob
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import time

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
def create_word_cloud(text_data, title):
    if not text_data:
        st.warning("No text data to create a word cloud.")
        return
    full_text = " ".join([t for t in text_data if t])
    if not full_text:
        st.warning("No valid text to create a word cloud.")
        return

    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'university', 'education', 'learning', 'student', 'technology', 'digital', 'institution',
                        'experience', 'use'}
    all_stopwords = stop_words.union(custom_stopwords)

    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=all_stopwords).generate(full_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=20)
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


def analyze_with_progress():
    """Show progress during analysis"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    steps = ["Loading data", "Processing text", "Analyzing sentiment", "Generating visualizations"]

    for i, step in enumerate(steps):
        progress_bar.progress((i + 1) / len(steps))
        status_text.text(f"⏳ {step}...")
        time.sleep(0.1)

    progress_bar.empty()
    status_text.text("✅ Analysis complete!")


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

df = None
text_cols = None
uploaded_file = None

# --- NEW: SIDEBAR NAVIGATION AND FILE UPLOADER ---
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

if df is not None:
    # --- NEW: FILTERING SECTION ---
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

    df_filtered = df.copy()
    if region_filter:
        df_filtered = df_filtered[df_filtered["Region"].isin(region_filter)]
    if country_filter:
        df_filtered = df_filtered[df_filtered["Country"].isin(country_filter)]
    if position_filter:
        df_filtered = df_filtered[df_filtered["Position"].isin(position_filter)]
    if institution_filter:
        df_filtered = df_filtered[df_filtered["Institution_Type"].isin(institution_filter)]

    st.sidebar.markdown(f"**Filtered Responses:** {len(df_filtered)} of {len(df)}")

    # --- NEW: TABBED INTERFACE FOR ANALYSIS SECTIONS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🔍 Question Analysis",
        "📈 Comparative Analysis",
        "📋 Data Explorer"
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
        create_word_cloud(responses, f"Most Frequent Words: {selected_label}")

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
                df_sent = df_filtered.copy()
                df_sent['Sentiment_Label'] = df_sent[selected_col_sentiment].apply(
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
            fig_emotion_sunburst = create_comparative_sunburst(df_filtered, selected_demographic, selected_col_emotion)
            st.plotly_chart(fig_emotion_sunburst, use_container_width=True)

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
st.caption("Built with ❤️ using Streamlit + VADER + Ekman Emotion Model | For research purposes only")