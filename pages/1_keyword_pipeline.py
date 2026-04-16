import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

st.title("🚀 Keyword Pipeline")
st.caption("Cluster → Classify → Score | Upload keywords → download file phân tier sẵn để viết bài")

# ─────────────────────────────────────────────────────────────
# NICHE/INTENT LABELS — khớp với training_data.csv
# ─────────────────────────────────────────────────────────────
# training_data niches: Food/Recipe, Garden/Outdoor, Hair/Beauty, Home Decor,
#                       Kitchen, Lifestyle, Other, Styling, Tattoo, Wedding/Craft
# training_data intents: diy-craft, food-baking, food-recipe, general, hair-beauty,
#                        outfit-style, pop-culture, product-specific, room-ideas, tattoo, wedding-event

NICHE_PROTOTYPES = {
    "Home Decor": [
        "bedroom decor ideas for small spaces",
        "living room furniture and decoration aesthetic",
        "bathroom vanity mirror wall art ideas",
        "entryway hallway rug shelf organization",
        "boho farmhouse modern minimalist interior design",
        "curtains drapes window treatment home styling",
        "throw pillow blanket cozy home aesthetic",
        "home organization storage basket bin declutter",
        "wall decor frame gallery art print",
        "couch sofa accent chair living room furniture",
        "chandelier pendant lamp sconce lighting",
    ],
    "Garden/Outdoor": [
        "backyard garden landscaping ideas design",
        "patio outdoor furniture seating decoration",
        "raised garden bed vegetable herb planting",
        "flower plant succulent indoor outdoor pot",
        "lawn care grass maintenance tips",
        "outdoor string lights patio ambiance",
        "garden tools planting watering hose",
        "pergola deck fence outdoor structure",
    ],
    "Kitchen": [
        "kitchen organization storage solutions countertop",
        "cookware pan pot set kitchen tools",
        "coffee maker espresso machine kitchen appliance",
        "knife set cutting board kitchen gadget",
        "kitchen shelf cabinet pantry organizer",
        "air fryer instant pot slow cooker pressure cooker",
        "kitchen decor aesthetic farmhouse modern",
        "toaster blender food processor small appliance",
    ],
    "Food/Recipe": [
        "easy chicken dinner recipe weeknight family",
        "pasta soup salad healthy meal prep ideas",
        "beef pork salmon seafood cooking dinner",
        "keto vegan vegetarian gluten free healthy eating",
        "crockpot slow cooker casserole one pot recipe",
        "chocolate cake cupcake dessert recipe from scratch",
        "cookie brownie bar baking easy beginner",
        "sourdough bread muffin scone pastry baking",
        "comfort food hearty filling dinner recipe",
        "meal prep batch cooking weekly plan",
    ],
    "Styling": [
        "outfit ideas what to wear casual everyday",
        "fashion style clothing aesthetic look inspiration",
        "dress jeans boots sneakers shoes styling",
        "capsule wardrobe minimalist fashion basics",
        "summer winter fall spring seasonal outfit",
        "bag purse handbag accessory jewelry styling",
        "how to style outfit ideas for women",
    ],
    "Hair/Beauty": [
        "hairstyle haircut hair color ideas inspiration",
        "blonde brunette highlights balayage color ideas",
        "nail art design manicure gel ideas",
        "skincare routine steps products morning night",
        "makeup tutorial look beginner natural glam",
        "curtain bangs layers pixie bob haircut",
        "lashes brows glam beauty look",
        "hair care treatment mask growth tips",
    ],
    "Tattoo": [
        "tattoo design ideas inspiration placement",
        "small fine line minimalist tattoo art",
        "sleeve floral geometric mandala tattoo",
        "meaningful symbol quote tattoo ideas",
        "tattoo aftercare healing moisturizer",
        "watercolor blackwork traditional tattoo style",
    ],
    "Wedding/Craft": [
        "wedding decoration ceremony reception ideas",
        "bridal shower bachelorette party ideas themes",
        "engagement proposal anniversary romantic ideas",
        "baby shower gender reveal party decoration",
        "birthday party table decoration theme setup",
        "wedding floral arrangement centerpiece bouquet",
        "DIY wedding craft decoration handmade budget",
        "crochet knitting sewing pattern handmade craft",
        "macrame wreath candle making craft project",
    ],
    "Lifestyle": [
        "morning routine productivity self care habits",
        "travel destination guide bucket list tips",
        "minimalist lifestyle wellness mental health",
        "personal finance budget saving money tips",
        "journal planner goal setting motivation",
    ],
    "Other": [
        "general ideas tips information guide list",
        "DIY home project tutorial step by step beginner",
        "woodworking build shelf furniture project",
        "sofa sectional couch living room furniture",
        "dining table chair set furniture ideas",
    ],
}

INTENT_PROTOTYPES = {
    "product-specific": [
        "best product comparison review top rated buy",
        "affordable quality product recommendation purchase",
        "top 10 picks under budget review",
        "where to buy product recommendation guide",
        "best comparison buying guide worth it",
    ],
    "room-ideas": [
        "bedroom ideas inspiration transformation small space",
        "living room design aesthetic mood board ideas",
        "bathroom makeover before after renovation reveal",
        "cozy aesthetic room setup ideas",
        "home tour interior design inspiration",
    ],
    "outfit-style": [
        "outfit of the day styling inspiration look",
        "what to wear casual date night occasion",
        "how to style layering mixing matching outfit",
        "aesthetic outfit ideas pinterest fashion",
    ],
    "food-recipe": [
        "easy recipe how to make step by step cooking",
        "dinner ideas quick recipe for the week",
        "healthy meal prep recipe collection batch",
        "recipe ingredients instructions method cooking",
    ],
    "food-baking": [
        "baking recipe from scratch tutorial beginner",
        "how to decorate cake cookie dessert frosting",
        "easy baking ideas project weekend",
        "baking tips techniques tricks tutorial",
    ],
    "diy-craft": [
        "DIY tutorial how to make craft project beginner",
        "step by step handmade homemade guide instructions",
        "easy weekend craft project activity",
        "make your own build tutorial materials needed",
    ],
    "hair-beauty": [
        "hair tutorial how to style at home easy",
        "makeup look tutorial step by step beginner",
        "skincare routine product recommendation review",
        "hair transformation before after color cut",
    ],
    "tattoo": [
        "tattoo ideas inspiration design gallery collection",
        "tattoo placement meaning symbolism ideas",
        "tattoo style guide what to choose",
    ],
    "wedding-event": [
        "wedding inspiration planning ideas checklist",
        "party decoration theme setup ideas DIY budget",
        "event planning tips ideas inspiration",
    ],
    "pop-culture": [
        "disney theme party decoration merchandise",
        "fandom gift ideas merchandise themed room",
        "movie show character themed decor",
    ],
    "general": [
        "ideas tips guide information list collection",
        "how to tips advice beginner guide",
    ],
}

STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","up","about","into","over","after","before","is","are","was",
    "were","be","been","have","has","do","does","did","will","would","could",
    "should","not","no","so","too","very","just","also","get","make","take",
    "your","my","our","their","this","that","these","those","what","which",
    "all","any","some","such","page","category","author","tag","post","blog",
}

SEP_RE   = re.compile(r"[-_]+")
YEAR_RE  = re.compile(r"\b(20\d{2}|19\d{2})\b")
ALPHA_RE = re.compile(r"[^a-z0-9 ]")

# ─────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Loading embedding model (lần đầu ~30s)...")
def load_st_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(show_spinner="📐 Building prototype embeddings...")
def build_prototypes(_model):
    def centroid(protos):
        embs = _model.encode(protos, show_progress_bar=False, convert_to_numpy=True)
        return np.mean(embs, axis=0)
    n_labels = list(NICHE_PROTOTYPES.keys())
    n_matrix = np.stack([centroid(NICHE_PROTOTYPES[k]) for k in n_labels])
    i_labels = list(INTENT_PROTOTYPES.keys())
    i_matrix = np.stack([centroid(INTENT_PROTOTYPES[k]) for k in i_labels])
    return n_labels, n_matrix, i_labels, i_matrix

# Cache model training để không retrain mỗi lần re-render
@st.cache_data(show_spinner=False)
def train_lgb_model(_tr_embs, _tr_niche, _tr_intent, _tr_wc, _y_tr):
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X_tr = np.hstack([_tr_embs, _tr_niche, _tr_intent, _tr_wc])

    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        max_depth=6, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, class_weight='balanced',
        random_state=42, verbose=-1, n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(lgb_clf, X_tr, _y_tr, cv=cv, scoring='roc_auc').mean()
    lgb_clf.fit(X_tr, _y_tr)
    return lgb_clf, float(auc)

def prob_to_tier(p):
    if p >= 70:   return 'Tier1_High'
    elif p >= 50: return 'Tier2_Medium'
    elif p >= 35: return 'Tier3_Low'
    else:         return 'Tier4_Skip'

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    st.subheader("Step 1 — Clustering")
    cluster_threshold = st.slider("Clustering Threshold", 0.70, 0.99, 0.82, 0.01,
        help="Cao hơn = cluster chặt hơn, ít keyword/cluster hơn")
    sim_filter = st.slider("Similarity Filter", 0.70, 0.99, 0.88, 0.01,
        help="Keyword dưới ngưỡng này sẽ tách thành singleton")
    st.subheader("Step 2 — Classify")
    overlap_threshold = st.slider("Niche Overlap Threshold", 0.05, 0.30, 0.15, 0.01,
        help="Gap < threshold → assign secondary niche")
    st.subheader("Step 3 — ML Score")
    batch_size = st.select_slider("Batch Size", options=[64, 128, 256], value=128)
    st.divider()
    st.markdown("**Hướng dẫn:**")
    st.markdown("1. Upload **keywords CSV** (cột `keyword`)")
    st.markdown("2. Training data tự load từ repo (hoặc upload thủ công)")
    st.markdown("3. Click **▶ Run Pipeline**")
    st.markdown("4. Download kết quả theo tier")

# ─────────────────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────────────────

col_kw, col_train = st.columns(2)

with col_kw:
    st.subheader("📂 Keywords cần score")
    kw_file = st.file_uploader(
        "Upload file keywords (.csv hoặc .xlsx)",
        type=["csv", "xlsx", "xls"],
        key="kw_upload"
    )
    if kw_file:
        try:
            preview = pd.read_csv(kw_file, nrows=5) if kw_file.name.endswith(".csv") else pd.read_excel(kw_file, nrows=5)
            st.success(f"✅ **{kw_file.name}**")
            st.dataframe(preview, use_container_width=True, height=150)
            kw_file.seek(0)
        except Exception as e:
            st.error(f"Lỗi: {e}")

with col_train:
    st.subheader("🧠 Training data (GA4)")
    # Thử load từ repo trước
    _default = Path(__file__).parent.parent / "data" / "training_data.csv"
    train_source = None

    if _default.exists():
        st.success("✅ **training_data.csv** (tự động từ repo)")
        st.dataframe(pd.read_csv(_default, nrows=5), use_container_width=True, height=150)
        train_source = "repo"
    else:
        train_upload = st.file_uploader(
            "Upload training_data.csv",
            type=["csv"], key="train_upload",
            help="Cần có cột: keyword, avg_ctr, niche, intent"
        )
        if train_upload:
            try:
                st.success(f"✅ **{train_upload.name}**")
                st.dataframe(pd.read_csv(train_upload, nrows=5), use_container_width=True, height=150)
                train_upload.seek(0)
                train_source = "upload"
            except Exception as e:
                st.error(f"Lỗi: {e}")
        else:
            st.info("Upload training_data.csv để bắt đầu")

# ─────────────────────────────────────────────────────────────
# RUN BUTTON
# ─────────────────────────────────────────────────────────────

st.divider()
can_run = kw_file is not None and train_source is not None
if not can_run:
    missing = []
    if kw_file is None: missing.append("keywords file")
    if train_source is None: missing.append("training data")
    st.info(f"👆 Cần upload: {', '.join(missing)}")

run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=True, disabled=not can_run)

# ─────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────

if run_btn and can_run:
    t_start = time.time()

    # ── 1. Load keywords ──────────────────────────────────────
    kw_file.seek(0)
    raw_kw = pd.read_csv(kw_file, dtype=str) if kw_file.name.endswith(".csv") else pd.read_excel(kw_file, dtype=str)
    raw_kw.columns = raw_kw.columns.str.strip()

    kw_col  = next((c for c in raw_kw.columns if any(k in c.lower() for k in ['keyword','kw','query','term','topic'])), raw_kw.columns[0])
    vol_col = next((c for c in raw_kw.columns if any(k in c.lower() for k in ['vol','volume','search','msv','sv'])), None)

    df = pd.DataFrame({'keyword': raw_kw[kw_col].astype(str).str.lower().str.strip()})
    df['volume'] = pd.to_numeric(raw_kw[vol_col], errors='coerce').fillna(0) if vol_col else 0
    df = df[df['keyword'].notna() & (df['keyword'].str.strip() != '') & (df['keyword'] != 'nan')]
    df = df.drop_duplicates('keyword').reset_index(drop=True)

    st.info(f"📊 **{len(df):,} unique keywords** | cột keyword: `{kw_col}` | volume: `{vol_col or 'không có'}`")

    # ── 2. Load training data ─────────────────────────────────
    if train_source == "repo":
        df_train = pd.read_csv(_default, encoding='utf-8-sig')
    else:
        train_upload.seek(0)
        df_train = pd.read_csv(train_upload, encoding='utf-8-sig')

    # Validate columns
    required = {'keyword', 'avg_ctr', 'niche', 'intent'}
    missing_cols = required - set(df_train.columns)
    if missing_cols:
        st.error(f"❌ Training data thiếu cột: {missing_cols}")
        st.stop()

    # Clean training data
    df_train = df_train[df_train['keyword'].notna()].copy()
    df_train['keyword'] = df_train['keyword'].astype(str).str.lower().str.strip()
    df_train = df_train[~df_train['keyword'].str.match(r'^\d+$|^page', na=False)]
    df_train['avg_ctr'] = pd.to_numeric(df_train['avg_ctr'], errors='coerce').fillna(0)
    df_train['is_converter'] = (df_train['avg_ctr'] > 0).astype(int)
    df_train = df_train.reset_index(drop=True)

    # Build label maps TỪ training data (ground truth)
    train_niches  = sorted(df_train['niche'].unique().tolist())
    train_intents = sorted(df_train['intent'].unique().tolist())
    niche_map  = {n: i for i, n in enumerate(train_niches)}
    intent_map = {n: i for i, n in enumerate(train_intents)}

    # ── Progress ──────────────────────────────────────────────
    progress = st.progress(0, text="Đang chuẩn bị...")
    status   = st.empty()

    # ── 3. Load model + Embed keywords ───────────────────────
    status.markdown("**⏳ Loading embedding model...**")
    model = load_st_model()
    n_labels, n_matrix, i_labels, i_matrix = build_prototypes(model)
    progress.progress(5, "Model loaded ✓")

    status.markdown(f"**⏳ Embedding {len(df):,} keywords...**")
    embeddings = model.encode(
        df['keyword'].tolist(), batch_size=batch_size,
        show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True,
    )
    progress.progress(25, "Embedding done ✓")

    # ══════════════════════════════════════════════════════════
    # STEP 1: CLUSTERING
    # ══════════════════════════════════════════════════════════
    status.markdown("**🔑 Step 1/3 — Clustering...**")
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    N = len(df)
    if N > 3000:
        from sklearn.neighbors import kneighbors_graph
        conn = kneighbors_graph(embeddings, n_neighbors=min(15, N-1),
                                metric='cosine', include_self=False, n_jobs=-1)
        cl = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='complete',
                                     distance_threshold=1-cluster_threshold, connectivity=conn)
    else:
        cl = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='complete',
                                     distance_threshold=1-cluster_threshold)
    labels = cl.fit_predict(embeddings)

    groups = {}
    for i, lbl in enumerate(labels):
        groups.setdefault(lbl, []).append(i)

    keywords = df['keyword'].tolist()
    volumes  = df['volume'].values
    cluster_main_map = {}
    cluster_size_map = {}
    similarity_map   = {}

    for lbl, idxs in groups.items():
        members = [(keywords[i], float(volumes[i])) for i in idxs]
        top3    = sorted(members, key=lambda x: x[1], reverse=True)[:3]
        main_kw = min(top3, key=lambda x: len(x[0]))[0]
        main_idx = idxs[next(j for j, (k, _) in enumerate(members) if k == main_kw)]

        # Tính similarity của các sub-keywords so với main
        sub_idxs = [i for i in idxs if keywords[i] != main_kw]
        sub_sims = {}
        if sub_idxs:
            sims = cos_sim(embeddings[main_idx].reshape(1, -1), embeddings[sub_idxs])[0]
            for si, sim in zip(sub_idxs, sims):
                sub_sims[keywords[si]] = float(sim)

        for i in idxs:
            kw  = keywords[i]
            sim = sub_sims.get(kw, 1.0)
            if kw != main_kw and sim < sim_filter:
                # Tách thành singleton
                cluster_main_map[kw] = kw
                cluster_size_map[kw] = 1
                similarity_map[kw]   = 100.0
            else:
                cluster_main_map[kw] = main_kw
                cluster_size_map[kw] = len(idxs)
                similarity_map[kw]   = round(sim * 100, 1) if kw != main_kw else 100.0

    df['cluster_main'] = df['keyword'].map(cluster_main_map)
    df['cluster_size'] = df['keyword'].map(cluster_size_map).fillna(1).astype(int)
    df['is_main']      = (df['keyword'] == df['cluster_main']).astype(int)
    df['similarity']   = df['keyword'].map(similarity_map)
    progress.progress(45, "Clustering done ✓")

    # ══════════════════════════════════════════════════════════
    # STEP 2: CLASSIFY → niche + intent
    # Dùng label names khớp với training_data
    # ══════════════════════════════════════════════════════════
    status.markdown("**🔍 Step 2/3 — Classifying niche + intent...**")
    niche_sims  = cos_sim(embeddings, n_matrix)
    intent_sims = cos_sim(embeddings, i_matrix)

    primary_niches = []; secondary_niches = []; niche_scores = []
    intents_col    = []; intent_scores    = []

    for i in range(len(df)):
        # Niche
        ns     = niche_sims[i]
        ranked = np.argsort(ns)[::-1]
        pn     = n_labels[ranked[0]]
        ps     = float(ns[ranked[0]])
        sn     = n_labels[ranked[1]] if len(ranked) > 1 and ps - float(ns[ranked[1]]) < overlap_threshold else ""

        # Intent
        ins   = intent_sims[i]
        top_i = int(np.argmax(ins))

        primary_niches.append(pn)
        secondary_niches.append(sn)
        niche_scores.append(round(ps, 4))
        intents_col.append(i_labels[top_i])
        intent_scores.append(round(float(ins[top_i]), 4))

    df['primary_niche']   = primary_niches
    df['secondary_niche'] = secondary_niches
    df['niche_score']     = niche_scores
    df['intent']          = intents_col
    df['intent_score']    = intent_scores
    progress.progress(65, "Classification done ✓")

    # ══════════════════════════════════════════════════════════
    # STEP 3: TRAIN MODEL + SCORE
    # Key fix: train_niches/intents từ training_data (ground truth)
    # Score keywords dùng niche/intent từ Step 2 (prototype classifier)
    # Cả hai đều dùng CÙNG label space từ training_data → map đúng
    # ══════════════════════════════════════════════════════════
    status.markdown("**🎯 Step 3/3 — Training ML model + Scoring...**")

    # Embed training keywords
    status.markdown("**🎯 Step 3/3 — Embedding training keywords...**")
    tr_embs = model.encode(
        df_train['keyword'].tolist(), batch_size=batch_size,
        show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True,
    )

    tr_niche_enc  = df_train['niche'].map(niche_map).fillna(-1).astype(int).values.reshape(-1, 1)
    tr_intent_enc = df_train['intent'].map(intent_map).fillna(-1).astype(int).values.reshape(-1, 1)
    tr_wc         = df_train['keyword'].str.split().str.len().values.reshape(-1, 1)
    y_tr          = df_train['is_converter'].values

    status.markdown("**🎯 Step 3/3 — Training LightGBM (5-fold CV)...**")
    lgb_clf, auc_mean = train_lgb_model(tr_embs, tr_niche_enc, tr_intent_enc, tr_wc, y_tr)

    # Score keywords mới — dùng niche/intent từ Step 2
    # Map về cùng integer encoding như training
    sc_niche_enc  = df['primary_niche'].map(niche_map).fillna(-1).astype(int).values.reshape(-1, 1)
    sc_intent_enc = df['intent'].map(intent_map).fillna(-1).astype(int).values.reshape(-1, 1)
    sc_wc         = df['keyword'].str.split().str.len().values.reshape(-1, 1)
    X_sc          = np.hstack([embeddings, sc_niche_enc, sc_intent_enc, sc_wc])

    proba = lgb_clf.predict_proba(X_sc)[:, 1]
    df['convert_prob'] = (proba * 100).round(1)
    df['tier']         = df['convert_prob'].apply(prob_to_tier)

    # Thống kê niche coverage (bao nhiêu % keywords có niche khớp training)
    matched = df['primary_niche'].isin(train_niches).mean() * 100

    progress.progress(100, "Pipeline complete ✓")
    status.empty()
    elapsed = time.time() - t_start

    # ══════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════
    st.success(
        f"✅ Hoàn thành trong **{elapsed:.0f}s** | "
        f"CV AUC: **{auc_mean:.4f}** | "
        f"Training: **{len(df_train):,}** keywords | "
        f"Niche match: **{matched:.0f}%**"
    )

    if matched < 80:
        st.warning(
            f"⚠️ Chỉ {matched:.0f}% keywords có niche khớp với training data. "
            f"Training niches: {', '.join(train_niches)}"
        )

    tier_counts = df['tier'].value_counts()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total",        f"{len(df):,}")
    c2.metric("🟢 Tier1",    f"{tier_counts.get('Tier1_High', 0):,}",   help="≥ 70%")
    c3.metric("🟡 Tier2",    f"{tier_counts.get('Tier2_Medium', 0):,}", help="50–70%")
    c4.metric("🟠 Tier3",    f"{tier_counts.get('Tier3_Low', 0):,}",    help="35–50%")
    c5.metric("🔴 Tier4 Skip",f"{tier_counts.get('Tier4_Skip', 0):,}",  help="< 35%")

    # Build output
    out_cols = ['keyword','volume','cluster_main','cluster_size','is_main','similarity',
                'primary_niche','secondary_niche','intent','convert_prob','tier',
                'niche_score','intent_score']
    out_cols = [c for c in out_cols if c in df.columns]
    df_out = df[out_cols].sort_values('convert_prob', ascending=False).reset_index(drop=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🟢 Tier1 — Làm ngay", "🟡 Tier2 — Tiềm năng", "⬇️ Download"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Niche distribution**")
            nd = df_out['primary_niche'].value_counts().reset_index()
            nd.columns = ['Niche', 'Keywords']
            st.dataframe(nd, use_container_width=True, hide_index=True)
        with col_b:
            st.write("**Intent distribution**")
            id_ = df_out['intent'].value_counts().reset_index()
            id_.columns = ['Intent', 'Keywords']
            st.dataframe(id_, use_container_width=True, hide_index=True)

        st.write("**Top 30 — highest convert probability**")
        st.dataframe(
            df_out.head(30)[['keyword','primary_niche','intent','convert_prob','tier','cluster_main','cluster_size']],
            use_container_width=True, hide_index=True,
        )

    with tab2:
        t1 = df_out[df_out['tier'] == 'Tier1_High']
        st.write(f"**{len(t1):,} keywords sẵn sàng để làm content** (convert_prob ≥ 70%)")
        st.dataframe(
            t1[['keyword','primary_niche','intent','convert_prob','cluster_main','cluster_size']],
            use_container_width=True, hide_index=True,
        )

    with tab3:
        t2 = df_out[df_out['tier'] == 'Tier2_Medium']
        st.write(f"**{len(t2):,} keywords tiềm năng** (50–70%)")
        st.dataframe(
            t2[['keyword','primary_niche','intent','convert_prob','cluster_main','cluster_size']],
            use_container_width=True, hide_index=True,
        )

    with tab4:
        now = datetime.now().strftime('%Y%m%d_%H%M')

        def to_csv_bytes(frame):
            return frame.to_csv(index=False).encode('utf-8')

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.download_button("⬇️ Full", data=to_csv_bytes(df_out),
                               file_name=f"pipeline_full_{now}.csv", mime="text/csv",
                               use_container_width=True)
        with d2:
            sub = df_out[df_out['tier'] == 'Tier1_High']
            st.download_button(f"🟢 Tier1 ({len(sub):,})", data=to_csv_bytes(sub),
                               file_name=f"Tier1_High_{now}.csv", mime="text/csv",
                               use_container_width=True)
        with d3:
            sub = df_out[df_out['tier'] == 'Tier2_Medium']
            st.download_button(f"🟡 Tier2 ({len(sub):,})", data=to_csv_bytes(sub),
                               file_name=f"Tier2_Medium_{now}.csv", mime="text/csv",
                               use_container_width=True)
        with d4:
            sub = df_out[df_out['tier'] == 'Tier3_Low']
            st.download_button(f"🟠 Tier3 ({len(sub):,})", data=to_csv_bytes(sub),
                               file_name=f"Tier3_Low_{now}.csv", mime="text/csv",
                               use_container_width=True)

        st.divider()
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df_out.to_excel(writer, sheet_name='All Keywords', index=False)
            for tier_label, sheet_name in [
                ('Tier1_High',   '🟢 Tier1 High'),
                ('Tier2_Medium', '🟡 Tier2 Medium'),
                ('Tier3_Low',    '🟠 Tier3 Low'),
                ('Tier4_Skip',   '🔴 Tier4 Skip'),
            ]:
                sub = df_out[df_out['tier'] == tier_label]
                if len(sub) > 0:
                    sub.to_excel(writer, sheet_name=sheet_name, index=False)
        buf.seek(0)
        st.download_button(
            "⬇️ Download Excel (multi-sheet)",
            data=buf,
            file_name=f"keyword_pipeline_{now}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True, type="primary",
        )
