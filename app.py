import streamlit as st

st.set_page_config(
    page_title="Keyword Pipeline",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Keyword Pipeline")
st.caption("Cluster → Classify → Score | All-in-one tool cho Pinterest affiliate portfolio")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔑 Step 1: Clustering")
    st.write("Gom nhóm keywords theo semantic similarity dùng all-MiniLM-L6-v2 + AgglomerativeClustering.")

with col2:
    st.subheader("🔍 Step 2: Classify")
    st.write("Tự động gán niche + intent cho từng keyword bằng prototype embeddings.")

with col3:
    st.subheader("🎯 Step 3: ML Score")
    st.write("Chấm điểm xác suất convert Amazon affiliate bằng LightGBM train từ data GA4 thực tế.")

st.divider()
st.info("👈 Chọn **Keyword Pipeline** từ sidebar để chạy toàn bộ pipeline, hoặc chạy từng tool riêng lẻ.")
st.caption("Model: all-MiniLM-L6-v2 + LightGBM · Trained on real GA4 data · Built for Pinterest affiliate portfolio")
