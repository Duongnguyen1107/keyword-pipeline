import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

st.title("🔑 Keyword Clustering")
st.caption("Powered by all-MiniLM-L6-v2 + AgglomerativeClustering")

with st.sidebar:
    st.header("⚙️ Settings")
    threshold  = st.slider("Clustering Threshold", 0.70, 0.99, 0.82, 0.01,
                           help="Cao hơn = cluster chặt hơn")
    sim_filter = st.slider("Similarity Filter",    0.70, 0.99, 0.88, 0.01,
                           help="Keyword dưới ngưỡng tách thành singleton")
    batch_size = st.select_slider("Batch Size", options=[32, 64, 128], value=64)
    st.divider()
    st.markdown("**Hướng dẫn:**")
    st.markdown("1. Upload file Excel/CSV")
    st.markdown("2. Điều chỉnh settings nếu cần")
    st.markdown("3. Click **Run Clustering**")
    st.markdown("4. Download kết quả")

uploaded = st.file_uploader("📂 Upload file keywords (.xlsx hoặc .csv)",
                             type=["xlsx","xls","csv"])

if uploaded:
    try:
        uploaded.seek(0)
        df_prev = pd.read_csv(uploaded, nrows=5) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded, nrows=5)
        st.success(f"✅ **{uploaded.name}**")
        st.dataframe(df_prev, use_container_width=True)
        uploaded.seek(0)
    except Exception as e:
        st.error(f"Lỗi: {e}"); st.stop()

    if st.button("🚀 Run Clustering", type="primary", use_container_width=True):
        uploaded.seek(0)
        df = pd.read_csv(uploaded, dtype=str) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded, dtype=str)
        df.columns = df.columns.str.strip()
        df = df.dropna(how='all')

        kw_col = next((c for c in df.columns if c.lower() in ('keyword','keywords','kw','key','query')), df.columns[0])
        vol_col = next((c for c in df.columns if any(x in c.lower() for x in ('vol','volume','search','msv','sv'))), None)

        keywords = df[kw_col].dropna().str.strip().tolist()
        keywords = list(dict.fromkeys([k for k in keywords if k]))
        volumes  = {}
        if vol_col:
            for _, row in df.iterrows():
                kw = str(row[kw_col]).strip() if pd.notna(row[kw_col]) else ''
                if not kw: continue
                try: v = int(str(row[vol_col]).replace(',','').replace('.','').strip())
                except: v = 0
                if kw not in volumes: volumes[kw] = v

        st.write(f"**{len(keywords):,} unique keywords** sẽ được clustering")

        with st.spinner("⏳ Loading model + encoding... (lần đầu ~30-60s)"):
            from sentence_transformers import SentenceTransformer
            @st.cache_resource
            def _load(): return SentenceTransformer('all-MiniLM-L6-v2')
            model = _load()
            embs = model.encode(keywords, batch_size=batch_size,
                                show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)

        with st.spinner("⏳ Clustering..."):
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics.pairwise import cosine_similarity
            N = len(keywords)
            if N > 3000:
                from sklearn.neighbors import kneighbors_graph
                conn = kneighbors_graph(embs, n_neighbors=min(15,N-1), metric='cosine', include_self=False, n_jobs=-1)
                cl = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='complete',
                                             distance_threshold=1-threshold, connectivity=conn)
            else:
                cl = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='complete',
                                             distance_threshold=1-threshold)
            labels = cl.fit_predict(embs)

        with st.spinner("⏳ Building output..."):
            groups = {}
            for i, lbl in enumerate(labels): groups.setdefault(lbl,[]).append(i)

            def pick_main(members_with_vol):
                if len(members_with_vol)==1: return members_with_vol[0][0]
                top3 = sorted(members_with_vol, key=lambda x:x[1], reverse=True)[:3]
                return min(top3, key=lambda x:len(x[0]))[0]

            rows = []
            for lbl, idxs in sorted(groups.items(), key=lambda x:-len(x[1])):
                members = [(keywords[i], volumes.get(keywords[i],0)) for i in idxs]
                cluster_vol = sum(v for _,v in members)
                main = pick_main(members)
                main_idx = idxs[next(j for j,(k,_) in enumerate(members) if k==main)]
                subs = [(keywords[i], volumes.get(keywords[i],0), i) for i in idxs if keywords[i]!=main]
                sim_scores = {}
                if subs:
                    sims = cosine_similarity(embs[main_idx].reshape(1,-1), embs[[i for _,_,i in subs]])[0]
                    for (kw,vol,_),sim in zip(subs,sims): sim_scores[kw]=sim
                rows.append({'Chu de chinh':main,'Tong Volume':cluster_vol,'Cluster Size':len(idxs),
                             'Keyword':main,'Volume':volumes.get(main,0),'Is Main':'YES','Similarity':'100%'})
                for kw,vol,_ in sorted(subs, key=lambda x:sim_scores.get(x[0],0), reverse=True):
                    sim = sim_scores.get(kw,0)
                    if sim < sim_filter:
                        rows.append({'Chu de chinh':kw,'Tong Volume':vol,'Cluster Size':1,
                                     'Keyword':kw,'Volume':vol,'Is Main':'YES','Similarity':'100%'})
                    else:
                        rows.append({'Chu de chinh':main,'Tong Volume':cluster_vol,'Cluster Size':len(idxs),
                                     'Keyword':kw,'Volume':vol,'Is Main':'NO','Similarity':f"{sim*100:.1f}%"})

            df_res = pd.DataFrame(rows).sort_values(['Tong Volume','Volume'],ascending=[False,False])

        n_cl  = len(set(labels))
        n_clu = sum(1 for lbl in labels if labels.tolist().count(lbl)>1)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Keywords",  f"{len(keywords):,}")
        c2.metric("Clusters",  f"{n_cl:,}")
        c3.metric("Clustered", f"{n_clu:,}")
        c4.metric("Coverage",  f"{n_clu/len(keywords)*100:.1f}%")

        st.subheader("📊 Preview (top 50 rows)")
        st.dataframe(df_res.head(50), use_container_width=True)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            df_res.to_excel(w, index=False)
        buf.seek(0)
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button("⬇️ Download Excel", data=buf,
                           file_name=f"Keyword_Clusters_{now}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           type="primary", use_container_width=True)
else:
    st.info("👆 Upload file keywords để bắt đầu")
