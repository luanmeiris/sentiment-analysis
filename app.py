import streamlit as st
from textblob import TextBlob
import pandas as pd
import plotly.express as px
from deep_translator import GoogleTranslator
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk

# Initial Configurations
nltk.download('punkt_tab')
st.set_page_config(page_title="Sentimenalyze", layout="wide")

# CSS Styling for a polished look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("Sentimenalyze: Data Intelligence")
st.sidebar.header("🛠️ Configurations")

# --- DATA INPUT ---
upload = st.sidebar.file_uploader("1. Upload a CSV or Excel file", type=["csv", "xlsx"])
text_input = st.sidebar.text_area("2. Or paste comments here:", placeholder="One per line...")

# Initialize session state to persist data during filtering
if 'df_resultado' not in st.session_state:
    st.session_state.df_resultado = None

if st.sidebar.button("Run Robust Analysis"):
    data = []
    if upload:
        try:
            df_input = pd.read_csv(upload) if upload.name.endswith('csv') else pd.read_excel(upload)
            # Assuming data is in the first column
            data = df_input.iloc[:, 0].astype(str).tolist()
        except Exception as e:
            st.error(f"Error reading file: {e}")
    elif text_input:
        data = [f.strip() for f in text_input.split('\n') if f.strip()]

    if data:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, f in enumerate(data):
            try:
                status_text.text(f"Analyzing sentence {i+1} of {len(data)}...")
                # Auto-translation to English for standardized analysis
                translation = GoogleTranslator(source='auto', target='en').translate(f)
                blob = TextBlob(translation)
                pol = blob.sentiment.polarity
                
                # Sentiment Classification
                if pol > 0.1: 
                    sent = "Positive"
                elif pol < -0.1: 
                    sent = "Negative"
                else: 
                    sent = "Neutral"
                
                results.append({"Original Text": f, "Sentiment": sent, "Score": pol})
                progress_bar.progress((i + 1) / len(data))
            except:
                continue
        
        st.session_state.df_resultado = pd.DataFrame(results)
        status_text.success("Analysis completed successfully!")
    else:
        st.sidebar.error("No data found for analysis!")

# --- DISPLAY AND FILTERS ---
if st.session_state.df_resultado is not None:
    df = st.session_state.df_resultado

    # Sidebar Filters
    st.sidebar.divider()
    st.sidebar.subheader("Filter View")
    sentiment_types = st.sidebar.multiselect(
        "Select sentiment types:",
        options=["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"]
    )
    
    # Apply Filters
    df_filtered = df[df['Sentiment'].isin(sentiment_types)]

    if not df_filtered.empty:
        # High-level Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Analyzed", len(df_filtered))
        m2.metric("Average Score", f"{df_filtered['Score'].mean():.2f}")
        m3.metric("Dominant Sentiment", df_filtered['Sentiment'].mode()[0])

        st.divider()

        # Dashboard Charts
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Sentiment Distribution")
            fig_pie = px.pie(df_filtered, names='Sentiment', color='Sentiment',
                             color_discrete_map={'Positive':'#2ecc71','Neutral':'#f1c40f','Negative':'#e74c3c'},
                             hole=0.5)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("Intensity Distribution (Score)")
            fig_hist = px.histogram(df_filtered, x="Score", nbins=15,
                                    color_discrete_sequence=['#3498db'],
                                    labels={'Score': 'Polarity Score (-1 to 1)'})
            st.plotly_chart(fig_hist, use_container_width=True)

        # Word Cloud
        st.subheader("Word Cloud (Frequent Terms)")
        text_total = " ".join(df_filtered['Original Text'])
        if text_total.strip():
            wordcloud = WordCloud(background_color="white", width=1200, height=400, colormap='viridis').generate(text_total)
            fig_wc, ax = plt.subplots(figsize=(15, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig_wc)

        # Data Table and Export
        st.divider()
        st.subheader("Detailed Data")
        st.dataframe(df_filtered, use_container_width=True)
        
        # O segredo está no 'utf-8-sig' e no 'index=False'
        csv = df_filtered.to_csv(index=False, sep=',').encode('utf-8-sig')

        st.download_button(
            label="📩 Download Filtered Report (CSV)",
            data=csv,
            file_name="sentiment_analysis_pro.csv",
            mime="text/csv"
        )
    else:
        st.warning("No data matches the selected filters.")
else:
    st.info("Waiting for data input to generate the professional dashboard...")