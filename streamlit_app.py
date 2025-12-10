"""
Streamlit Chat UI for SupplyChainGPT
"""

import streamlit as st
import requests
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

# Page config
st.set_page_config(
    page_title="SupplyChainGPT",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
    }
    .source-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1E88E5;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 2px;
    }
    .badge-warning { background-color: #fff3cd; color: #856404; }
    .badge-info { background-color: #d1ecf1; color: #0c5460; }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_sources" not in st.session_state:
        st.session_state.current_sources = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = "default_user"


def call_api(endpoint: str, method: str = "GET", data: dict = None, files: dict = None):
    """Make API call"""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        if method == "GET":
            response = requests.get(url, params=data, timeout=30)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, data=data, timeout=60)
            else:
                response = requests.post(url, json=data, timeout=60)
        else:
            return None

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the backend is running.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def render_sidebar():
    """Render sidebar with filters and document upload"""
    with st.sidebar:
        st.markdown("### Filters")

        # Quick filters
        warehouse = st.selectbox(
            "Warehouse",
            ["All Warehouses", "WH-001", "WH-002", "WH-003"],
            key="warehouse_filter"
        )

        sku = st.text_input("SKU ID", placeholder="e.g., SKU-12345", key="sku_filter")

        st.markdown("---")

        # Document upload
        st.markdown("### Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "xlsx", "csv", "txt"],
            key="file_uploader"
        )

        if uploaded_file is not None:
            doc_type = st.selectbox(
                "Document Type",
                ["report", "policy", "sop", "contract", "manual", "export"],
                key="doc_type"
            )

            if st.button("Upload", key="upload_btn"):
                with st.spinner("Uploading and processing..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    data = {"doc_type": doc_type}
                    result = call_api("documents/upload", "POST", data=data, files=files)

                    if result and result.get("status") == "success":
                        st.success(f"Uploaded! {result.get('chunks_created', 0)} chunks created.")
                    else:
                        st.error("Upload failed")

        st.markdown("---")

        # System stats
        st.markdown("### System Stats")
        if st.button("Refresh Stats", key="refresh_stats"):
            stats = call_api("documents/stats")
            if stats:
                st.metric("Total Chunks", stats.get("total_chunks", 0))

        st.markdown("---")

        # Forecast section
        st.markdown("### Quick Forecast")
        forecast_sku = st.text_input("SKU for Forecast", placeholder="SKU-12345", key="forecast_sku")

        if st.button("Get Forecast", key="forecast_btn") and forecast_sku:
            with st.spinner("Generating forecast..."):
                result = call_api(
                    "forecast",
                    "POST",
                    data={
                        "sku_id": forecast_sku,
                        "periods": 30,
                        "include_safety_stock": True
                    }
                )
                if result:
                    st.session_state.forecast_result = result


def render_chat_interface():
    """Render main chat interface"""
    st.markdown('<p class="main-header">SupplyChainGPT</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-Powered Supply Chain Assistant</p>',
        unsafe_allow_html=True
    )

    # Chat history
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])

                    # Show badges if any
                    if message.get("badges"):
                        badges_html = ""
                        for badge in message["badges"]:
                            badges_html += f'<span class="badge badge-warning">{badge}</span>'
                        st.markdown(badges_html, unsafe_allow_html=True)

                    # Show confidence
                    if message.get("confidence"):
                        conf = message["confidence"]
                        conf_class = "high" if conf > 0.7 else "medium" if conf > 0.4 else "low"
                        st.markdown(
                            f'<small class="confidence-{conf_class}">Confidence: {conf:.0%}</small>',
                            unsafe_allow_html=True
                        )

    # Chat input
    query = st.chat_input("Ask about inventory, forecasts, or policies...")

    if query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})

        # Get response
        with st.spinner("Thinking..."):
            # Check if forecast is needed
            include_forecast = any(
                term in query.lower()
                for term in ["forecast", "predict", "demand", "safety stock", "reorder"]
            )

            # Extract SKU if mentioned
            sku_id = None
            if st.session_state.get("sku_filter"):
                sku_id = st.session_state.sku_filter

            response = call_api(
                "ask/simple",
                "POST",
                data={
                    "query": query,
                    "user_id": st.session_state.user_id,
                    "include_forecast": include_forecast,
                    "sku_id": sku_id
                }
            )

            if response:
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.get("answer", "I couldn't generate a response."),
                    "confidence": response.get("confidence", 0),
                    "badges": response.get("warning_badges", []),
                    "citations": response.get("citations", [])
                })

                # Update sources
                st.session_state.current_sources = response.get("citations", [])

        st.rerun()


def render_sources_panel():
    """Render sources panel"""
    st.markdown("### Sources")

    if not st.session_state.current_sources:
        st.info("Ask a question to see relevant sources")
        return

    for i, source in enumerate(st.session_state.current_sources, 1):
        with st.expander(f"📄 {source.get('doc_title', 'Unknown')} ({source.get('relevance_score', 0):.2f})"):
            st.markdown(f"**Source:** {source.get('source_uri', 'N/A')}")
            st.markdown("**Excerpt:**")
            st.markdown(f"> {source.get('chunk_text', 'No content')[:300]}...")


def render_forecast_chart():
    """Render forecast visualization if available"""
    if "forecast_result" not in st.session_state:
        return

    result = st.session_state.forecast_result
    forecast_data = result.get("forecast", [])

    if not forecast_data:
        return

    st.markdown("### Demand Forecast")

    # Create dataframe
    df = pd.DataFrame(forecast_data)
    df["date"] = pd.to_datetime(df["date"])

    # Create chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["predicted_demand"],
        mode="lines+markers",
        name="Predicted Demand",
        line=dict(color="#1E88E5")
    ))

    # Add confidence bounds if available
    if "lower_bound" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["upper_bound"],
            mode="lines",
            name="Upper Bound",
            line=dict(dash="dash", color="#90CAF9"),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["lower_bound"],
            mode="lines",
            name="Lower Bound",
            line=dict(dash="dash", color="#90CAF9"),
            fill="tonexty",
            fillcolor="rgba(30, 136, 229, 0.1)",
            showlegend=False
        ))

    fig.update_layout(
        title=f"Demand Forecast for {result.get('sku_id', 'SKU')}",
        xaxis_title="Date",
        yaxis_title="Demand (units)",
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show safety stock info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Safety Stock", f"{result.get('safety_stock', 0):.0f} units")
    with col2:
        st.metric("Reorder Point", f"{result.get('reorder_point', 0):.0f} units")
    with col3:
        st.metric("Model", result.get("model_used", "N/A"))


def main():
    """Main application"""
    init_session_state()

    # Layout
    render_sidebar()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        render_chat_interface()

        # Show forecast chart if available
        render_forecast_chart()

    with col2:
        render_sources_panel()

        # Quick actions
        st.markdown("### Quick Actions")

        if st.button("Clear Chat", key="clear_chat"):
            st.session_state.messages = []
            st.session_state.current_sources = []
            st.rerun()

        if st.button("Export Conversation", key="export_chat"):
            # Create export data
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "messages": st.session_state.messages,
                "sources": st.session_state.current_sources
            }
            st.download_button(
                "Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()
