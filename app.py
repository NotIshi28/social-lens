import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from wordcloud import WordCloud
from textblob import TextBlob
import nltk
from collections import Counter
import plotly.subplots as sp
import re


try:
    nltk.data.find("vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")
    nltk.download("punkt")


st.set_page_config(
    page_title="Social Media Analytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def show_data_format_guide():
    st.markdown("""
    ### Required CSV Format Guide
    
    Your CSV file should include these columns:
    
    #### Essential Columns:
    - `timestamp`: Post date and time
    - `id`: Unique post identifier
    - `caption`: Post text content
    - `like_count`: Number of likes
    - `comment_count`: Number of comments
    - `share_count`: Number of shares
    - `save_count`: Number of saves
    
    #### Engagement Columns:
    - `reach`: Post reach count
    - `impressions`: Post impression count
    - `engagement_type`: Type of post (photo/video/carousel)
    - `video_views`: Number of video views
    - `video_duration`: Video length in seconds
    
    #### Audience Columns:
    - `audience_age_group`: Age distribution (format: "18-24: 30%, 25-34: 45%, 35-44: 25%")
    - `audience_gender`: Gender distribution (format: "Male: 45%, Female: 55%")
    - `location`: Post location
    - `device_type`: Device used (mobile/desktop/tablet)
    
    #### Additional Metrics:
    - `time_spent`: Average time spent on post
    - `link_clicks`: Number of link clicks
    - `profile_visits`: Profile visits from post
    - `follower_count`: Account follower count at post time
    
    [Download Sample Template CSV](https://github.com/NotIshi28/social-lens/sample_data.csv)
    """)


@st.cache_data
def load_data(file):
    try:
        if isinstance(file, str):
            data = pd.read_csv(file)
        else:
            data = pd.read_csv(file)

        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data["id_short"] = data["id"].astype(str).str[-4:]

        data["engagement_rate"] = (
            (
                data["like_count"]
                + data["comment_count"]
                + data["share_count"]
                + data["save_count"]
            )
            / data["follower_count"]
            * 100
        )

        data["age_data"] = data["audience_age_group"].apply(
            lambda x: {
                group.split(":")[0].strip(): float(
                    group.split(":")[1].strip().rstrip("%")
                )
                for group in x.split(",")
            }
        )

        data["gender_data"] = data["audience_gender"].apply(
            lambda x: {
                group.split(":")[0].strip(): float(
                    group.split(":")[1].strip().rstrip("%")
                )
                for group in x.split(",")
            }
        )

        data["sentiment_scores"] = data["caption"].apply(analyze_sentiment)
        data["sentiment_polarity"] = data["sentiment_scores"].apply(
            lambda x: x["polarity"]
        )
        data["sentiment_subjectivity"] = data["sentiment_scores"].apply(
            lambda x: x["subjectivity"]
        )

        data["hashtags_list"] = data["caption"].apply(extract_hashtags)
        data["mentions_list"] = data["caption"].apply(extract_mentions)

        data["hour"] = data["timestamp"].dt.hour
        data["day_of_week"] = data["timestamp"].dt.day_name()
        data["month"] = data["timestamp"].dt.month_name()

        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()


def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    blob = TextBlob(str(text))
    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
    }


def extract_hashtags(text):
    """Extract hashtags from text"""
    return re.findall(r"#(\w+)", str(text))


def extract_mentions(text):
    """Extract mentions from text"""
    return re.findall(r"@(\w+)", str(text))


def create_engagement_metrics_chart(data):
    metrics_df = pd.melt(
        data,
        id_vars=["timestamp"],
        value_vars=["like_count", "comment_count", "share_count", "save_count"],
        var_name="Metric",
        value_name="Count",
    )

    fig = px.line(
        metrics_df,
        x="timestamp",
        y="Count",
        color="Metric",
        title="Engagement Metrics Over Time",
    )
    return fig


def create_content_performance_analysis(data):
    avg_by_type = (
        data.groupby("engagement_type")
        .agg(
            {
                "like_count": "mean",
                "comment_count": "mean",
                "share_count": "mean",
                "engagement_rate": "mean",
            }
        )
        .round(2)
    )

    fig = px.bar(
        avg_by_type, title="Average Performance by Content Type", barmode="group"
    )
    return fig


def create_audience_demographics(data):

    age_data = pd.DataFrame([dict(x) for x in data["age_data"]])
    age_means = age_data.mean()

    fig_age = px.pie(
        values=age_means.values,
        names=age_means.index,
        title="Audience Age Distribution",
    )

    gender_data = pd.DataFrame([dict(x) for x in data["gender_data"]])
    gender_means = gender_data.mean()

    fig_gender = px.pie(
        values=gender_means.values,
        names=gender_means.index,
        title="Audience Gender Distribution",
    )

    return fig_age, fig_gender


def create_geographic_analysis(data):
    location_counts = data["location"].value_counts()
    fig = px.bar(
        x=location_counts.index,
        y=location_counts.values,
        title="Content Distribution by Location",
    )
    return fig


def create_device_analysis(data):
    device_counts = data["device_type"].value_counts()
    fig = px.pie(
        values=device_counts.values,
        names=device_counts.index,
        title="Device Type Distribution",
    )
    return fig


def create_video_performance_analysis(data):
    video_data = data[data["engagement_type"] == "video"]
    fig = px.scatter(
        video_data,
        x="video_duration",
        y="video_views",
        size="engagement_rate",
        title="Video Performance Analysis",
    )
    return fig


def format_numbers(value):
    if isinstance(value, float):
        return f"{value:.2f}%"
    return f"{value:,}"


def create_sentiment_analysis_charts(data):

    fig = sp.make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Sentiment Distribution",
            "Sentiment vs. Engagement",
            "Sentiment Over Time",
            "Sentiment by Content Type",
        ),
    )

    fig.add_trace(
        go.Histogram(x=data["sentiment_polarity"], name="Sentiment Distribution"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data["sentiment_polarity"],
            y=data["engagement_rate"],
            mode="markers",
            name="Engagement Rate",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["sentiment_polarity"],
            mode="lines+markers",
            name="Sentiment Trend",
        ),
        row=2,
        col=1,
    )

    avg_sentiment = data.groupby("engagement_type")["sentiment_polarity"].mean()
    fig.add_trace(
        go.Bar(
            x=avg_sentiment.index, y=avg_sentiment.values, name="Avg Sentiment by Type"
        ),
        row=2,
        col=2,
    )

    fig.update_layout(height=800, showlegend=False)
    return fig


def create_advanced_insights(data):
    """Generate advanced insights from the data"""
    insights = []

    best_hour = data.groupby("hour")["engagement_rate"].mean().idxmax()
    best_day = data.groupby("day_of_week")["engagement_rate"].mean().idxmax()

    insights.append(f"🕒 Best posting time: {best_hour}:00 on {best_day}")

    best_type = data.groupby("engagement_type")["engagement_rate"].mean().idxmax()
    insights.append(f"📊 Most engaging content type: {best_type}")

    avg_sentiment = data["sentiment_polarity"].mean()
    sentiment_impact = data["sentiment_polarity"].corr(data["engagement_rate"])

    insights.append(
        f"😊 Average content sentiment: {'Positive' if avg_sentiment > 0 else 'Negative'}"
    )
    insights.append(f"💫 Sentiment correlation with engagement: {sentiment_impact:.2f}")

    return insights


def main():
    st.title("📊 Social Lens: Analytics Dashboard")

    st.sidebar.header("📁 Data Source")
    data_source = st.sidebar.radio(
        "Select Data Source", ["Use Sample Data", "Upload Custom Data"]
    )

    if data_source == "Upload Custom Data":
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            st.sidebar.success("✅ File uploaded successfully!")
        else:
            st.sidebar.info("ℹ️ Please upload a CSV file")
            if st.sidebar.button("Show Data Format Guide"):
                show_data_format_guide()
            st.stop()
    else:
        try:
            data = load_data("sample_data.csv")
        except FileNotFoundError:
            st.error("❌ Sample data file not found. Please upload your own data.")
            st.stop()

    st.sidebar.header("🎯 Filters")

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(data["timestamp"].min(), data["timestamp"].max()),
        min_value=data["timestamp"].min().date(),
        max_value=data["timestamp"].max().date(),
    )

    content_types = st.sidebar.multiselect(
        "Content Type",
        options=data["engagement_type"].unique(),
        default=data["engagement_type"].unique(),
    )

    with st.sidebar.expander("Advanced Filters"):
        min_engagement = st.slider(
            "Minimum Engagement Rate",
            min_value=0.0,
            max_value=float(data["engagement_rate"].max()),
            value=0.0,
        )

        sentiment_filter = st.select_slider(
            "Sentiment Filter",
            options=["Negative", "Neutral", "Positive"],
            value=("Negative", "Positive"),
        )

    filtered_data = data[
        (data["timestamp"].dt.date >= date_range[0])
        & (data["timestamp"].dt.date <= date_range[1])
        & (data["engagement_type"].isin(content_types))
        & (data["engagement_rate"] >= min_engagement)
    ]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📈 Overview",
            "😊 Sentiment Analysis",
            "👥 Audience Insights",
            "🎥 Content Analysis",
            "📱 Platform & Device",
            "🔍 Advanced Insights",
        ]
    )

    with tab1:

        st.subheader("📊 Quick Stats")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_engagement = (
                filtered_data["like_count"].sum() + filtered_data["comment_count"].sum()
            )
            st.metric(
                "Total Engagement",
                f"{total_engagement:,}",
                delta=f"{(total_engagement/len(filtered_data)):,.0f} per post",
            )
        with col2:
            avg_er = filtered_data["engagement_rate"].mean()
            st.metric("Avg Engagement Rate", f"{avg_er:.2f}%")
        with col3:
            total_reach = filtered_data["reach"].sum()
            st.metric("Total Reach", f"{total_reach:,}")
        with col4:
            avg_sentiment = filtered_data["sentiment_polarity"].mean()
            st.metric(
                "Sentiment Score",
                f"{avg_sentiment:.2f}",
                delta="Positive" if avg_sentiment > 0 else "Negative",
            )

        st.plotly_chart(
            create_engagement_metrics_chart(filtered_data), use_container_width=True
        )

        st.plotly_chart(
            create_content_performance_analysis(filtered_data), use_container_width=True
        )

    with tab2:
        st.subheader("🎭 Sentiment Analysis")

        sentiment_fig = create_sentiment_analysis_charts(filtered_data)
        st.plotly_chart(sentiment_fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            positive_text = " ".join(
                filtered_data[filtered_data["sentiment_polarity"] > 0]["caption"]
            )
            positive_wordcloud = WordCloud(
                background_color="white", width=800, height=400
            ).generate(positive_text)
            st.image(
                positive_wordcloud.to_array(), caption="Positive Content Word Cloud"
            )

        with col2:
            negative_text = " ".join(
                filtered_data[filtered_data["sentiment_polarity"] < 0]["caption"]
            )
            negative_wordcloud = WordCloud(
                background_color="white", width=800, height=400
            ).generate(negative_text)
            st.image(
                negative_wordcloud.to_array(), caption="Negative Content Word Cloud"
            )

    with tab3:
        st.subheader("👥 Audience Analysis")

        col1, col2 = st.columns(2)
        fig_age, fig_gender = create_audience_demographics(filtered_data)

        with col1:
            st.plotly_chart(fig_age, use_container_width=True)
        with col2:
            st.plotly_chart(fig_gender, use_container_width=True)

        st.plotly_chart(
            create_geographic_analysis(filtered_data), use_container_width=True
        )

        growth_fig = px.line(
            filtered_data,
            x="timestamp",
            y="follower_count",
            title="Follower Growth Over Time",
        )
        st.plotly_chart(growth_fig, use_container_width=True)

    with tab4:
        st.subheader("🎥 Content Analysis")

        st.plotly_chart(
            create_video_performance_analysis(filtered_data), use_container_width=True
        )

        col1, col2 = st.columns(2)
        with col1:
            hashtag_counts = Counter(
                [tag for tags in filtered_data["hashtags_list"] for tag in tags]
            )
            top_hashtags = pd.DataFrame(
                hashtag_counts.most_common(10), columns=["Hashtag", "Count"]
            )
            st.plotly_chart(
                px.bar(top_hashtags, x="Hashtag", y="Count", title="Top Hashtags")
            )

        with col2:
            mention_counts = Counter(
                [
                    mention
                    for mentions in filtered_data["mentions_list"]
                    for mention in mentions
                ]
            )
            top_mentions = pd.DataFrame(
                mention_counts.most_common(10), columns=["Mention", "Count"]
            )
            st.plotly_chart(
                px.bar(top_mentions, x="Mention", y="Count", title="Top Mentions")
            )

    with tab5:
        st.subheader("📱 Platform & Device Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                create_device_analysis(filtered_data), use_container_width=True
            )

        with col2:
            time_analysis = px.histogram(
                filtered_data,
                x="hour",
                title="Posts by Hour of Day",
                labels={"hour": "Hour of Day"},
            )
            st.plotly_chart(time_analysis, use_container_width=True)

    with tab6:
        st.subheader("🔍 Advanced Insights")

        insights = create_advanced_insights(filtered_data)
        for insight in insights:
            st.info(insight)

        correlation_data = filtered_data[
            [
                "like_count",
                "comment_count",
                "share_count",
                "save_count",
                "reach",
                "impressions",
                "sentiment_polarity",
            ]
        ].corr()

        st.write("### 🔄 Metric Correlations")
        fig = px.imshow(
            correlation_data,
            title="Metric Correlation Heatmap",
            color_continuous_scale="RdBu",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.sidebar.markdown("---")
    export_format = st.sidebar.selectbox("Export Format", ["CSV", "Excel", "JSON"])

    if st.sidebar.button("Export Analysis"):
        if export_format == "CSV":
            data = filtered_data.to_csv(index=False)
            mime = "text/csv"
            file_extension = "csv"
        elif export_format == "Excel":
            buffer = io.BytesIO()
            filtered_data.to_excel(buffer, index=False)
            data = buffer.getvalue()
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            file_extension = "xlsx"
        else:
            data = filtered_data.to_json(orient="records")
            mime = "application/json"
            file_extension = "json"

        st.sidebar.download_button(
            label="Download Data",
            data=data,
            file_name=f"social_media_analysis.{file_extension}",
            mime=mime,
        )


if __name__ == "__main__":
    main()
