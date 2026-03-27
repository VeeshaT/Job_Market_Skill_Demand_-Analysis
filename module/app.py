import streamlit as st
import pandas as pd
import plotly.express as px
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from modules.data_loader import DataEngine
from modules.constants import TOPIC_STOPWORDS

# 1. CONFIG & CSS THEME
st.set_page_config(page_title="X Analytics", layout="wide")

def inject_x_theme():
    st.markdown("""
    <style>
        /* --- 1. GLOBAL & HEADER --- */
        .stApp {
            background-color: #000000 !important;
            color: #E7E9EA !important;
        }
        
        header[data-testid="stHeader"] {
            background-color: rgba(0,0,0,0) !important;
            color: #E7E9EA !important;
        }
        header[data-testid="stHeader"] button {
            color: #E7E9EA !important;
        }

        /* --- 2. SIDEBAR --- */
        section[data-testid="stSidebar"] {
            background-color: #000000 !important;
            border-right: 1px solid #2F3336;
        }
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] label {
            color: #E7E9EA !important;
        }

        [data-testid="stSidebar"] div.stButton > button {
            background-color: #2F3336 !important; 
            color: #E7E9EA !important;
            border: 1px solid #536471 !important;
        }
        [data-testid="stSidebar"] div.stButton > button:hover {
            border-color: #F91880 !important;
            color: #F91880 !important;
        }

        [data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
            background-color: #1D9BF0 !important; 
            color: white !important;
            border: none !important;
        }
        [data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover {
            background-color: #1A8CD8 !important;
        }

        /* --- 3. TOMBOL & INPUT --- */
        div.stButton > button {
            background-color: #1D9BF0 !important;
            color: #ffffff !important;
            border-radius: 9999px !important;
            border: none !important;
            font-weight: 700 !important;
            padding: 0.5rem 1.5rem !important;
        }
        div.stButton > button:hover {
            background-color: #1A8CD8 !important;
        }

        /* Tombol Browse & Download */
        [data-testid="stFileUploader"] small {
                color: #E7E9EA !important;
        }        
        [data-testid="stFileUploader"] button {
            background-color: #1D9BF0 !important;
            color: #ffffff !important;
            border: none !important;
        }
        [data-testid="stDownloadButton"] button {
            background-color: #2F3336 !important;
            color: #E7E9EA !important;
            border: 1px solid #536471 !important;
            border-radius: 9999px !important;
        }
        [data-testid="stDownloadButton"] button:hover {
            border-color: #1D9BF0 !important;
            color: #1D9BF0 !important;
        }

        /* Input Fields */
        [data-testid="stFileUploader"] section, 
        div[data-baseweb="select"] > div, 
        div[data-baseweb="input"] > div {
            background-color: #202327 !important;
            border: 1px solid #2F3336 !important;
            color: #E7E9EA !important;
            border-radius: 8px;
        }
        input { color: #E7E9EA !important; }

        /* --- 4. DROPDOWN --- */
        div[data-baseweb="select"] > div {
            background-color: #202327 !important;
            border: 1 px solid #2F3336 !important;
            color: #E7E9EA !important;
        }
        div[data-baseweb="popover"] div, 
        div[data-baseweb="menu"] div,
        li[role="option"] {
            background-color: #000000 !important;
            color: #E7E9EA !important;
        }
        li[role="option"]:hover, li[aria-selected="true"] {
            # background-color: #1D9BF0 !important;
            color: #ffffff !important;
        }

        /* --- 5. TABEL --- */
        div[data-testid="stDataFrame"] {
            background-color: #000000 !important;
            border: 1px solid #2F3336 !important;
            border-radius: 10px;
        }
        div[data-testid="stDataFrame"] div[role="columnheader"] {
            background-color: #202327 !important;
            color: #E7E9EA !important;
            border-bottom: 1px solid #2F3336 !important;
        }
        div[data-testid="stDataFrame"] div[role="gridcell"] {
            background-color: #000000 !important;
            color: #E7E9EA !important;
        }

        /* --- 6. METRIC CARDS --- */
        div[data-testid="stMetric"] {
            background-color: #000000;
            border: 1px solid #2F3336;
            padding: 15px;
            border-radius: 15px;
        }
        div[data-testid="stMetric"] label { color: #71767B !important; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #E7E9EA !important; }

        /* --- 7. TEXT & TABS --- */
        h1, h2, h3, p, li, .stMarkdown { color: #E7E9EA !important; }
        
        button[data-baseweb="tab"] { color: #71767B !important; }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #E7E9EA !important;
            border-bottom: 2px solid #1D9BF0 !important;
        }

    </style>
    """, unsafe_allow_html=True)

# Helper: Style Chart
def style_chart(fig):
    fig.update_layout(
        template='plotly_dark', 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E7E9EA'),
        margin=dict(t=40, l=0, r=0, b=0),
        legend=dict(font=dict(color='#E7E9EA'))
    )
    fig.update_xaxes(gridcolor='#2F3336', zerolinecolor='#2F3336', title_font=dict(color='#E7E9EA'), tickfont=dict(color='#71767B'))
    fig.update_yaxes(gridcolor='#2F3336', zerolinecolor='#2F3336', title_font=dict(color='#E7E9EA'), tickfont=dict(color='#71767B'))
    return fig

# Helper: Wordcloud
def render_dark_wc(text, colormap='Blues', title="WordCloud"):
    if not text:
        st.info("Data tidak cukup.")
        return
    st.subheader(title)
    wc = WordCloud(width=800, height=400, background_color='black', 
                   colormap=colormap, stopwords=TOPIC_STOPWORDS).generate(text)
    fig, ax = plt.subplots(facecolor='black')
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# 2. MAIN APPLICATION
def main():
    inject_x_theme() # Load CSS
    st.sidebar.title("𝕏 Analytics")
    st.sidebar.markdown("---")
    menu = st.sidebar.radio("Menu", ["🏠 Beranda", "📊 Sentimen", "📈 Tren Waktu", "🗂️ Topik Isu", "⭐ Interaksi", "📢 Iklan, Spam, & Bot"])
    
    st.sidebar.markdown("---")
    
    # 1. Bagian Input (Data Source)
    st.sidebar.subheader("📂 Dataset")
    st.sidebar.caption("Direkomendasikan scrape dengan Tweet-Harvest")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (Tweet-Harvest)", type="csv", label_visibility="collapsed")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(base_dir, 'data', 'gojek.csv')

    final_file = None
    is_demo = False

    if uploaded_file:
        final_file = uploaded_file
        st.sidebar.success("✅ Menggunakan Data User")
    elif os.path.exists(default_path):
        final_file = default_path
        is_demo = True
        st.sidebar.info("ℹ️ Mode Demo: Data Gojek")
    else:
        st.sidebar.error("⚠️ File data tidak ditemukan.")
        st.stop()

    df = DataEngine.process_dataset(final_file)

    # 2. Bagian Actions
    if df is not None:
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ Actions")

        col1, col2 = st.sidebar.columns([1, 1])
        with col1:
            if st.button("🔄 Reset", use_container_width=True):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()     
        with col2:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh",
                data=csv_data,
                file_name='hasil_analisis.csv',
                mime='text/csv',
                use_container_width=True
            )
        
        df_organic = df[df['is_promo'] == False]
        df_promo = df[df['is_promo'] == True]

        # 1. BERANDA
        if menu == "🏠 Beranda":
            st.title("Dashboard Summary")
            
            total_organic = len(df_organic)
            total_neg = len(df_organic[df_organic['sentiment'] == 'Negatif'])
            neg_rate = (total_neg / total_organic * 100) if total_organic > 0 else 0
            avg_eng = df_organic['engagement_score'].mean()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Percakapan", f"{total_organic:,}")
            c2.metric("Sentimen Negatif", f"{neg_rate:.1f}%")
            c3.metric("Avg. Engagement", f"{avg_eng:.1f}")
            c4.metric("Spam Terbuang", f"{len(df_promo):,}")
            
            st.markdown("---")
            
            col_left, col_right = st.columns([2, 1])
            with col_left:
                st.subheader("📈 Tren Tweet 7 Hari Terakhir")
                daily_trend = df_organic.groupby('date_only').size().reset_index(name='counts')
                daily_trend['date_only'] = pd.to_datetime(daily_trend['date_only'])
                fig_trend = px.area(daily_trend, x='date_only', y='counts')
                fig_trend.update_traces(line_color='#1D9BF0', fillcolor='rgba(29, 155, 240, 0.3)')
                st.plotly_chart(style_chart(fig_trend), use_container_width=True)
                
            with col_right:
                st.subheader("🔥 Top Words")
                try:
                    vec = CountVectorizer(max_features=10, stop_words=list(TOPIC_STOPWORDS), ngram_range=(1, 1))
                    X = vec.fit_transform(df_organic['clean_text'].dropna())
                    sum_words = X.sum(axis=0) 
                    words_freq = sorted([(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()], key=lambda x: x[1], reverse=True)
                    df_wc = pd.DataFrame(words_freq, columns=['Kata', 'Frekuensi'])
                    fig_bar = px.bar(df_wc, x='Frekuensi', y='Kata', orientation='h', color='Frekuensi', color_continuous_scale='Blues')
                    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(style_chart(fig_bar), use_container_width=True)
                except:
                    st.info("Data belum cukup.")

            if not df_organic.empty:
                top_tweet = df_organic.loc[df_organic['engagement_score'].idxmax()]
                st.markdown("### 🌟 Sorotan: Tweet Viral")
                st.markdown(f"""
                <div style="border: 1px solid #2F3336; padding: 20px; border-radius: 12px; background-color: #000000; margin-bottom: 20px;">
                    <div style="color: #71767B; font-size: 14px; margin-bottom: 5px;">@{top_tweet['username']} • {top_tweet['date_only']}</div>
                    <div style="font-size: 18px; color: #E7E9EA; margin-bottom: 15px;">"{top_tweet['full_text']}"</div>
                    <div style="color: #1D9BF0; font-weight: bold;">
                        ❤️ {int(top_tweet['favorite_count'])} &nbsp;&nbsp; 🔁 {int(top_tweet['retweet_count'])} &nbsp;&nbsp; 💬 {int(top_tweet['reply_count'])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # 2. SENTIMEN
        elif menu == "📊 Sentimen":
            st.title("Analisis Sentimen")
            c1, c2 = st.columns([1, 2])
            with c1:
                counts = df_organic['sentiment'].value_counts().reset_index()
                counts.columns = ['Sentimen', 'Jumlah']
                fig = px.pie(counts, values='Jumlah', names='Sentimen', hole=0.5,
                             color='Sentimen', color_discrete_map={'Positif':'#00BA7C', 'Negatif':'#F91880', 'Netral': '#1D9BF0'})
                st.plotly_chart(style_chart(fig), use_container_width=True)
            with c2:
                st.subheader("📖 Baca Tweet")
                pil = st.selectbox("Filter Kategori:", ["Semua", "Negatif", "Positif", "Netral"])
                view = df_organic if pil == "Semua" else df_organic[df_organic['sentiment'] == pil]
                view = view.head(20) 
                
                st.caption(f"Menampilkan {len(view)} tweet terbaru:")
                with st.container(height=600):
                    for idx, row in view.iterrows():
                        border_color = "#1D9BF0" 
                        if row['sentiment'] == 'Positif': border_color = "#00BA7C"
                        elif row['sentiment'] == 'Negatif': border_color = "#F91880"
                        
                        st.markdown(f"""
                        <div style="border: 1px solid {border_color}; padding: 15px; border-radius: 12px; margin-bottom: 10px; background-color: #000000;">
                            <div style="display: flex; justify-content: space-between;">
                                <strong style="color: #E7E9EA">@{row['username']}</strong>
                                <span style="color: {border_color}; font-size: 12px; border: 1px solid {border_color}; padding: 2px 8px; border-radius: 10px;">{row['sentiment']}</span>
                            </div>
                            <p style="margin-top: 8px; color: #E7E9EA; font-size: 15px;">{row['full_text']}</p>
                            <div style="color: #71767B; font-size: 13px; margin-top: 10px;">
                                📅 {row['date_only']} &nbsp;•&nbsp; ❤️ {row['favorite_count']} 🔁 {row['retweet_count']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
            wc_col1, wc_col2 = st.columns(2)
            with wc_col1:
                text_pos = " ".join(df_organic[df_organic['sentiment']=='Positif']['clean_text'])
                render_dark_wc(text_pos, 'Greens', "☁️ Kata-kata Positif")
            with wc_col2:
                text_neg = " ".join(df_organic[df_organic['sentiment']=='Negatif']['clean_text'])
                render_dark_wc(text_neg, 'Reds', "☁️ Kata-kata Negatif")

        # 3. TREN WAKTU
        elif menu == "📈 Tren Waktu":
            st.title("Tren Aktivitas")
            daily = df_organic.groupby('date_only').agg({'id_str': 'count', 'sentiment_score': 'mean'}).reset_index()
            daily['date_only'] = pd.to_datetime(daily['date_only'])
            daily = daily.sort_values('date_only')
            
            st.subheader("Volume Tweet Harian")
            fig_vol = px.line(daily, x='date_only', y='id_str', markers=True)
            fig_vol.update_traces(line_color='#1D9BF0')
            st.plotly_chart(style_chart(fig_vol), use_container_width=True)
            
            st.subheader("Rata-rata Sentimen Harian")
            fig_sent = px.bar(daily, x='date_only', y='sentiment_score',
                              color='sentiment_score', color_continuous_scale=['#F91880', '#00BA7C'])
            st.plotly_chart(style_chart(fig_sent), use_container_width=True)

        # 4. TOPIK ISU
        elif menu == "🗂️ Topik Isu":
            st.title("Clustering Isu")
            if len(df_organic) > 0:
                k = st.slider("Jumlah Topik", 2, 8, 4)
                if st.button("Mulai Analisis"):
                    with st.spinner("Clustering..."):
                        vec = TfidfVectorizer(max_features=500, max_df=0.25, min_df=3, ngram_range=(1, 2))
                        try:
                            X = vec.fit_transform(df_organic['clean_text'].dropna())
                            km = KMeans(n_clusters=k, random_state=42).fit(X)
                            vocab = vec.get_feature_names_out()
                            
                            cols = st.columns(2)
                            for i in range(k):
                                center = km.cluster_centers_[i]
                                top_words = [vocab[x] for x in center.argsort()[-10:][::-1]]
                                with cols[i%2]:
                                    st.markdown(f"""
                                    <div style="border:1px solid #2F3336; padding:15px; border-radius:10px; margin-bottom:10px; background-color:#000000;">
                                        <h4 style="color:#1D9BF0; margin:0;">Topik {i+1}</h4>
                                        <p style="color:#E7E9EA; font-size:14px; margin-top:5px;">{", ".join(top_words)}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            st.markdown("---")
                            st.subheader("📊 Seberapa Dominan Tiap Topik?")
                            df_organic['topic_cluster'] = km.labels_
                            topic_counts = df_organic['topic_cluster'].value_counts().sort_index().reset_index()
                            topic_counts.columns = ['Topik', 'Jumlah']
                            topic_counts['Topik'] = topic_counts['Topik'].apply(lambda x: f"Topik {x+1}")
                            
                            fig_topic = px.bar(topic_counts, x='Topik', y='Jumlah', color='Topik')
                            st.plotly_chart(style_chart(fig_topic), use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error: {e}")

        # 5. INTERAKSI
        elif menu == "⭐ Interaksi":
            st.title("Analisis Viralitas")
            if len(df_organic) > 0:
                st.subheader("📈 Sebaran Tweet Viral")
                st.info("Engagement score = Likes + (2 x Retweets) + (1.5 x Replies)")
                df_organic['short_text'] = df_organic['full_text'].str.slice(0, 100) + "..."
                fig_viral = px.scatter(df_organic, x='created_at', y='engagement_score', size='engagement_score', color='sentiment',
                                     hover_data=['username', 'short_text'], 
                                     color_discrete_map={'Positif':'#00CC96', 'Negatif':'#EF553B', 'Netral':'#636EFA'})
                st.plotly_chart(style_chart(fig_viral), use_container_width=True)
                
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.subheader("🏆 Top 10 Tweet")
                    top_tweets = df_organic.nlargest(10, 'engagement_score')
                    with st.container(height=600):
                        for _, row in top_tweets.iterrows():
                            st.markdown(f"""
                            <div style="border: 1px solid #2F3336; padding: 15px; border-radius: 12px; margin-bottom: 10px; background-color: #000000;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <strong style="color: #E7E9EA">@{row['username']}</strong>
                                        <span style="color: #71767B; font-size: 12px;"> • {row['date_only']}</span>
                                    </div>
                                    <div style="background-color: rgba(29, 155, 240, 0.1); border: 1px solid #1D9BF0; padding: 2px 8px; border-radius: 10px;">
                                        <strong style="color: #1D9BF0; font-size: 12px;">🏆 Skor: {int(row['engagement_score'])}</strong>
                                    </div>
                                </div>
                                <p style="margin-top: 8px; color: #E7E9EA; font-size: 14px;">{row['full_text']}</p>
                                <div style="color: #71767B; font-size: 12px; margin-top: 5px;">
                                    ❤️ {int(row['favorite_count'])} &nbsp; 🔁 {int(row['retweet_count'])} &nbsp; 💬 {int(row['reply_count'])}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                with col_b:
                    st.subheader("👑 Top Akun")
                    top_users = df_organic.groupby('username')['engagement_score'].sum().nlargest(5).reset_index()
                    fig_users = px.bar(top_users, x='engagement_score', y='username', orientation='h')
                    fig_users.update_traces(marker_color='#1D9BF0')
                    st.plotly_chart(style_chart(fig_users), use_container_width=True)

        # 6. SPAM
        elif menu == "📢 Iklan, Spam, & Bot":
            st.title("Monitor Spam")
            if len(df_promo) == 0:
                st.success("Bersih dari spam.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Spam", len(df_promo))
                c2.metric("Akun Bot", df_promo['username'].nunique())
                c3.metric("Interaksi Palsu", int(df_promo['engagement_score'].sum()))
                
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.subheader("Top Spammers")
                    top_spammers = df_promo['username'].value_counts().nlargest(10).reset_index()
                    fig_spam = px.bar(top_spammers, x='count', y='username', orientation='h')
                    fig_spam.update_traces(marker_color='#F91880')
                    st.plotly_chart(style_chart(fig_spam), use_container_width=True)
                with col_s2:
                    st.subheader("Isi Spam")
                    text_spam = " ".join(df_promo['full_text'].astype(str))
                    render_dark_wc(text_spam, 'inferno', "☁️ Isi Konten Spam")
                
                st.subheader("🕒 Jam Operasional Bot")
                df_promo['hour'] = df_promo['created_at'].dt.hour
                hourly = df_promo['hour'].value_counts().sort_index().reset_index()
                hourly.columns = ['Jam', 'Jumlah Spam']
                
                fig_time = px.area(hourly, x='Jam', y='Jumlah Spam', markers=True)
                fig_time.update_traces(line_color='#F91880', fillcolor='rgba(249, 24, 128, 0.3)')
                st.plotly_chart(style_chart(fig_time), use_container_width=True)

                st.subheader("50 Tweet Spam")
                with st.container(height=500):
                    spam_view = df_promo.head(50)
                    for _, row in spam_view.iterrows():
                        st.markdown(f"""
                        <div style="border: 1px solid #F91880; padding: 12px; border-radius: 10px; margin-bottom: 10px; background-color: rgba(249, 24, 128, 0.05);">
                            <div style="display: flex; justify-content: space-between;">
                                <strong style="color: #E7E9EA">@{row['username']}</strong>
                                <span style="color: #F91880; font-size: 11px; border: 1px solid #F91880; padding: 2px 6px; border-radius: 8px;">SPAM DETECTED</span>
                            </div>
                            <p style="margin-top: 5px; color: #E7E9EA; font-size: 14px;">{row['full_text']}</p>
                            <div style="color: #71767B; font-size: 12px;">🕒 {row['created_at']}</div>
                        </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
