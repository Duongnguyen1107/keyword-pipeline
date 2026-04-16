#!/usr/bin/env python3
"""
keyword_pipeline.py — Pinterest Affiliate Keyword Pipeline
===========================================================
Gộp 3 tools thành 1 pipeline liên hoàn:
  Step 1: Keyword Clustering    (semantic grouping)
  Step 2: URL/Keyword Classifier (niche + intent)
  Step 3: ML Scorer             (convert_prob + tier)

INPUT:
  - keywords_to_score.csv  : file keyword thô (cột: keyword, optional: volume)
  - training_data.csv      : data GA4 để train model (cột: keyword, avg_ctr, niche, intent)
  - model.pkl              : model đã train (nếu có, bỏ qua bước train)

OUTPUT (folder output/):
  - pipeline_full.csv          : tất cả keywords với đủ thông tin
  - pipeline_Tier1_High.csv    : làm content ngay
  - pipeline_Tier2_Medium.csv  : tiềm năng
  - pipeline_Tier3_Low.csv     : ưu tiên sau
  - pipeline_summary.txt       : báo cáo tóm tắt

COMMANDS:
  # Full pipeline (train + cluster + classify + score)
  python keyword_pipeline.py run \\
    --keywords keywords_to_score.csv \\
    --training training_data.csv \\
    --output output/

  # Dùng model đã train (bỏ qua bước train)
  python keyword_pipeline.py run \\
    --keywords keywords_to_score.csv \\
    --model model.pkl \\
    --output output/

  # Chỉ train/update model
  python keyword_pipeline.py train \\
    --training training_data.csv \\
    --model model.pkl

  # Xem thông tin model
  python keyword_pipeline.py info --model model.pkl

REQUIREMENTS:
  pip install sentence-transformers lightgbm scikit-learn pandas numpy openpyxl
"""

import argparse
import pickle
import re
import sys
import time
import warnings
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONSTANTS — SHARED ACROSS ALL STEPS
# ─────────────────────────────────────────────────────────────

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

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
        "dresser nightstand bed frame bedroom furniture",
    ],
    "Garden/Outdoor": [
        "backyard garden landscaping ideas design",
        "patio outdoor furniture seating decoration",
        "raised garden bed vegetable herb planting",
        "flower plant succulent indoor outdoor pot",
        "lawn care grass maintenance tips",
        "greenhouse herb garden growing seeds",
        "outdoor string lights patio ambiance",
        "garden tools planting watering hose",
        "pergola deck fence outdoor structure",
        "bird feeder wildlife nature garden",
    ],
    "Kitchen": [
        "kitchen organization storage solutions countertop",
        "cookware pan pot set kitchen tools",
        "coffee maker espresso machine kitchen appliance",
        "knife set cutting board kitchen gadget",
        "kitchen shelf cabinet pantry organizer",
        "air fryer instant pot slow cooker pressure cooker",
        "kitchen decor aesthetic farmhouse modern",
        "dish rack utensil holder kitchen accessories",
        "toaster blender food processor small appliance",
    ],
    "Food/Recipe": [
        "easy chicken dinner recipe weeknight family",
        "pasta soup salad healthy meal prep ideas",
        "beef pork salmon seafood cooking dinner",
        "keto vegan vegetarian gluten free healthy eating",
        "crockpot slow cooker casserole one pot recipe",
        "sauce marinade dressing seasoning homemade",
        "30 minute quick easy dinner ideas",
        "meal prep batch cooking weekly plan",
        "comfort food hearty filling dinner recipe",
    ],
    "Food/Baking": [
        "chocolate cake cupcake dessert recipe from scratch",
        "cookie brownie bar baking easy beginner",
        "sourdough bread muffin scone pastry baking",
        "frosting buttercream icing fondant cake decorating",
        "cheesecake no bake dessert easy recipe",
        "birthday cake ideas decoration tutorial",
        "holiday Christmas Easter baking treats",
        "pie tart galette pastry shell filling",
    ],
    "Styling": [
        "outfit ideas what to wear casual everyday",
        "fashion style clothing aesthetic look inspiration",
        "dress jeans boots sneakers shoes styling",
        "capsule wardrobe minimalist fashion basics",
        "summer winter fall spring seasonal outfit",
        "bag purse handbag accessory jewelry styling",
        "body type flattering clothes styling tips",
        "street style trendy fashion look",
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
        "perfume fragrance beauty recommendation",
    ],
    "Tattoo": [
        "tattoo design ideas inspiration placement",
        "small fine line minimalist tattoo art",
        "sleeve floral geometric mandala tattoo",
        "meaningful symbol quote tattoo ideas",
        "tattoo aftercare healing moisturizer",
        "watercolor blackwork traditional tattoo style",
        "feminine delicate tattoo ideas for women",
    ],
    "Wedding/Event": [
        "wedding decoration ceremony reception ideas",
        "bridal shower bachelorette party ideas themes",
        "engagement proposal anniversary romantic ideas",
        "baby shower gender reveal party decoration",
        "birthday party table decoration theme setup",
        "wedding floral arrangement centerpiece bouquet",
        "DIY wedding craft decoration handmade budget",
        "wedding dress bridesmaid gown style",
        "wedding favor gift guest table seating",
    ],
    "DIY/Craft": [
        "DIY home project tutorial step by step beginner",
        "crochet knitting sewing pattern handmade craft",
        "woodworking build shelf furniture project",
        "painting canvas art craft kids activity",
        "repurpose upcycle thrift flip makeover project",
        "resin pour clay pottery craft tutorial",
        "macrame wreath candle making craft project",
    ],
    "Lifestyle": [
        "morning routine productivity self care habits",
        "travel destination guide bucket list tips",
        "minimalist lifestyle wellness mental health",
        "cozy reading book recommendation hobby",
        "personal finance budget saving money tips",
        "journal planner goal setting motivation",
    ],
    "Furniture": [
        "sofa sectional couch living room furniture",
        "dining table chair set furniture ideas",
        "bookcase bookshelf storage unit furniture",
        "bed frame headboard platform furniture bedroom",
        "outdoor patio furniture set lounge chair",
        "accent furniture side table console entry",
        "affordable budget furniture home setup",
    ],
}

INTENT_PROTOTYPES = {
    "product-specific": [
        "best rug comparison review top rated buy",
        "affordable quality product recommendation purchase",
        "top 10 picks under budget review",
        "where to buy product recommendation guide",
        "product review honest opinion worth it",
        "best air fryer comparison buying guide",
        "cheap affordable budget option recommendation",
    ],
    "room-ideas": [
        "bedroom ideas inspiration transformation small space",
        "living room design aesthetic mood board ideas",
        "bathroom makeover before after renovation reveal",
        "apartment rental decor ideas no damage",
        "home tour interior design room inspiration",
        "cozy aesthetic room setup ideas",
    ],
    "outfit-style": [
        "outfit of the day styling inspiration look",
        "what to wear casual date night occasion",
        "how to style layering mixing matching outfit",
        "aesthetic outfit ideas pinterest fashion",
        "complete look head to toe styling",
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
        "movie show character themed decor room",
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
    "www","com","net","org","admin",
}

SEP_RE   = re.compile(r"[-_]+")
YEAR_RE  = re.compile(r"\b(20\d{2}|19\d{2})\b")
ALPHA_RE = re.compile(r"[^a-z0-9 ]")

# ─────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")


def load_st_model():
    log(f"Loading embedding model: {EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list, st_model, batch_size: int = 256) -> np.ndarray:
    log(f"Embedding {len(texts):,} texts...")
    t0 = time.time()
    vecs = st_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    log(f"Done in {time.time()-t0:.1f}s")
    return vecs


def clean_keyword(text: str) -> str:
    """Normalize keyword text — giống clean_slug nhưng dành cho keyword phrases."""
    text = SEP_RE.sub(" ", str(text))
    text = YEAR_RE.sub("", text)
    text = ALPHA_RE.sub("", text.lower()).strip()
    words = text.split()
    while words and words[0] in STOP_WORDS:
        words.pop(0)
    while words and words[-1] in STOP_WORDS:
        words.pop()
    return " ".join(words) or text


def build_prototype_matrix(st_model, prototypes: dict) -> tuple:
    """Build centroid embeddings cho mỗi niche/intent."""
    labels = list(prototypes.keys())
    centroids = []
    for label in labels:
        embs = st_model.encode(
            prototypes[label],
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        centroids.append(np.mean(embs, axis=0))
    return labels, np.stack(centroids)


def prob_to_tier(p: float) -> str:
    if p >= 70:   return 'Tier1_High'
    elif p >= 50: return 'Tier2_Medium'
    elif p >= 35: return 'Tier3_Low'
    else:         return 'Tier4_Skip'


# ─────────────────────────────────────────────────────────────
# STEP 1: CLUSTERING
# ─────────────────────────────────────────────────────────────

def step_cluster(df: pd.DataFrame, embeddings: np.ndarray,
                 threshold: float = 0.82,
                 sim_filter: float = 0.88) -> pd.DataFrame:
    """
    Cluster keywords bằng AgglomerativeClustering.
    Trả về df với thêm cột: cluster_id, cluster_main, cluster_size, similarity
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    log(f"Clustering {len(df):,} keywords (threshold={threshold})...")

    keywords = df['keyword'].tolist()
    N = len(keywords)
    distance_threshold = 1 - threshold

    if N > 3000:
        from sklearn.neighbors import kneighbors_graph
        connectivity = kneighbors_graph(
            embeddings, n_neighbors=min(15, N - 1),
            metric='cosine', include_self=False, n_jobs=-1,
        )
        clustering = AgglomerativeClustering(
            n_clusters=None, metric='cosine', linkage='complete',
            distance_threshold=distance_threshold, connectivity=connectivity,
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None, metric='cosine', linkage='complete',
            distance_threshold=distance_threshold,
        )

    labels = clustering.fit_predict(embeddings)

    # Build groups
    groups = {}
    for i, lbl in enumerate(labels):
        groups.setdefault(lbl, []).append(i)

    # Chọn representative
    volumes = df.get('volume', pd.Series(0, index=df.index))

    cluster_id_map   = {}  # keyword → cluster_id
    cluster_main_map = {}  # keyword → cluster_main keyword
    cluster_size_map = {}  # cluster_id → size
    similarity_map   = {}  # keyword → similarity to main (%)

    for lbl, idxs in groups.items():
        members = [(keywords[i], float(volumes.iloc[i]) if i < len(volumes) else 0) for i in idxs]
        # Pick representative: highest volume top-3, then shortest
        top3 = sorted(members, key=lambda x: x[1], reverse=True)[:3]
        main_kw = min(top3, key=lambda x: len(x[0]))[0]
        main_idx = idxs[next(j for j, (k, _) in enumerate(members) if k == main_kw)]
        main_emb = embeddings[main_idx].reshape(1, -1)

        sub_idxs = [i for i in idxs if keywords[i] != main_kw]
        sub_sims = {}
        if sub_idxs:
            sub_embs = embeddings[sub_idxs]
            sims = cos_sim(main_emb, sub_embs)[0]
            for si, sim in zip(sub_idxs, sims):
                sub_sims[keywords[si]] = float(sim)

        for i in idxs:
            kw = keywords[i]
            sim = sub_sims.get(kw, 1.0)

            # Nếu similarity thấp hơn sim_filter → tách thành singleton
            if kw != main_kw and sim < sim_filter:
                cluster_id_map[kw]   = f"solo_{i}"
                cluster_main_map[kw] = kw
                cluster_size_map[f"solo_{i}"] = 1
                similarity_map[kw]   = 100.0
            else:
                cluster_id_map[kw]   = str(lbl)
                cluster_main_map[kw] = main_kw
                cluster_size_map[str(lbl)] = len(idxs)
                similarity_map[kw]   = round(sim * 100, 1) if kw != main_kw else 100.0

    df = df.copy()
    df['cluster_id']   = df['keyword'].map(cluster_id_map)
    df['cluster_main'] = df['keyword'].map(cluster_main_map)
    df['cluster_size'] = df['cluster_id'].map(cluster_size_map).fillna(1).astype(int)
    df['similarity']   = df['keyword'].map(similarity_map)
    df['is_main']      = (df['keyword'] == df['cluster_main']).astype(int)

    n_clusters = df['cluster_id'].nunique()
    clustered  = (df['cluster_size'] > 1).sum()
    log(f"Clustering done: {n_clusters:,} clusters | {clustered:,} keywords clustered ({clustered/len(df)*100:.1f}%)")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 2: CLASSIFY (niche + intent)
# ─────────────────────────────────────────────────────────────

def step_classify(df: pd.DataFrame, embeddings: np.ndarray,
                  st_model,
                  overlap_threshold: float = 0.15) -> pd.DataFrame:
    """
    Classify mỗi keyword → primary_niche, secondary_niche, intent
    bằng cosine similarity với prototype centroids.
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    log("Building prototype embeddings...")
    niche_labels, niche_matrix   = build_prototype_matrix(st_model, NICHE_PROTOTYPES)
    intent_labels, intent_matrix = build_prototype_matrix(st_model, INTENT_PROTOTYPES)

    log(f"Classifying {len(df):,} keywords → niche + intent...")
    niche_sims  = cos_sim(embeddings, niche_matrix)
    intent_sims = cos_sim(embeddings, intent_matrix)

    primary_niches   = []
    secondary_niches = []
    niche_scores     = []
    intents          = []
    intent_scores    = []

    for i in range(len(df)):
        # Niche
        ns     = niche_sims[i]
        ranked = np.argsort(ns)[::-1]
        pn     = niche_labels[ranked[0]]
        ps     = float(ns[ranked[0]])
        sn     = ""
        if len(ranked) > 1 and ps - float(ns[ranked[1]]) < overlap_threshold:
            sn = niche_labels[ranked[1]]

        # Intent
        ins   = intent_sims[i]
        top_i = int(np.argmax(ins))

        primary_niches.append(pn)
        secondary_niches.append(sn)
        niche_scores.append(round(ps, 4))
        intents.append(intent_labels[top_i])
        intent_scores.append(round(float(ins[top_i]), 4))

    df = df.copy()
    df['primary_niche']   = primary_niches
    df['secondary_niche'] = secondary_niches
    df['niche_score']     = niche_scores
    df['intent']          = intents
    df['intent_score']    = intent_scores

    log(f"Classification done | Top niche: {df['primary_niche'].value_counts().index[0]}")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 3: TRAIN ML MODEL
# ─────────────────────────────────────────────────────────────

def train_model(training_path: str, model_path: str, st_model) -> dict:
    """Train LightGBM từ training_data.csv → lưu model.pkl."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
    from sklearn.metrics import classification_report

    log(f"Loading training data: {training_path}")
    df = pd.read_csv(training_path, encoding='utf-8-sig')

    required = {'keyword', 'avg_ctr', 'niche', 'intent'}
    missing  = required - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns in training data: {missing}")
        sys.exit(1)

    # Clean
    df = df[df['keyword'].notna() & (df['keyword'].astype(str).str.strip() != '')]
    df = df[~df['keyword'].astype(str).str.match(r'^\d+$|^page', na=False)]
    df['keyword'] = df['keyword'].astype(str).str.lower().str.strip()
    df = df.reset_index(drop=True)
    df['is_converter'] = (df['avg_ctr'] > 0).astype(int)

    log(f"Training data: {len(df):,} keywords | {df['is_converter'].mean()*100:.1f}% converters | "
        f"{df['niche'].nunique()} niches | {df['intent'].nunique()} intents")

    niches  = sorted(df['niche'].unique().tolist())
    intents = sorted(df['intent'].unique().tolist())
    niche_map  = {n: i for i, n in enumerate(niches)}
    intent_map = {n: i for i, n in enumerate(intents)}

    # Features
    embeddings = embed(df['keyword'].tolist(), st_model)
    niche_enc  = df['niche'].map(niche_map).fillna(-1).astype(int).values.reshape(-1, 1)
    intent_enc = df['intent'].map(intent_map).fillna(-1).astype(int).values.reshape(-1, 1)
    word_count = df['keyword'].str.split().str.len().values.reshape(-1, 1)
    X = np.hstack([embeddings, niche_enc, intent_enc, word_count])
    y = df['is_converter'].values

    log(f"Feature matrix: {X.shape}")

    # CV
    log("Cross-validating (5-fold)...")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        max_depth=6, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, class_weight='balanced',
        random_state=42, verbose=-1, n_jobs=-1,
    )
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(lgb_clf, X, y, cv=cv, scoring='roc_auc')
    f1  = cross_val_score(lgb_clf, X, y, cv=cv, scoring='f1')
    log(f"CV ROC-AUC: {auc.mean():.4f} ± {auc.std():.4f} | F1: {f1.mean():.4f} ± {f1.std():.4f}")

    y_pred = cross_val_predict(lgb_clf, X, y, cv=cv)
    print(classification_report(y, y_pred, target_names=['No Convert', 'Convert']))

    # Final fit
    log("Fitting final model on full dataset...")
    lgb_clf.fit(X, y)

    payload = {
        'model':      lgb_clf,
        'niche_map':  niche_map,
        'intent_map': intent_map,
        'auc_cv':     float(auc.mean()),
        'f1_cv':      float(f1.mean()),
        'trained_on': len(df),
        'niches':     niches,
        'intents':    intents,
    }

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(payload, f)
    size_mb = Path(model_path).stat().st_size / 1024 / 1024
    log(f"Model saved: {model_path} ({size_mb:.1f} MB)")
    return payload


# ─────────────────────────────────────────────────────────────
# STEP 3: SCORE
# ─────────────────────────────────────────────────────────────

def step_score(df: pd.DataFrame, embeddings: np.ndarray, payload: dict) -> pd.DataFrame:
    """
    Score keywords bằng LightGBM model.
    df phải có cột: keyword, primary_niche, intent (từ step_classify)
    """
    lgb_clf    = payload['model']
    niche_map  = payload['niche_map']
    intent_map = payload['intent_map']

    niche_enc  = df['primary_niche'].map(niche_map).fillna(-1).astype(int).values.reshape(-1, 1)
    intent_enc = df['intent'].map(intent_map).fillna(-1).astype(int).values.reshape(-1, 1)
    word_count = df['keyword'].str.split().str.len().values.reshape(-1, 1)
    X = np.hstack([embeddings, niche_enc, intent_enc, word_count])

    log(f"Scoring {len(df):,} keywords...")
    proba = lgb_clf.predict_proba(X)[:, 1]

    df = df.copy()
    df['convert_prob'] = (proba * 100).round(1)
    df['tier']         = df['convert_prob'].apply(prob_to_tier)

    log("Scoring done")
    return df


# ─────────────────────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────

def run_pipeline(args):
    t_start = time.time()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load keywords ─────────────────────────────────────────
    log(f"Loading keywords: {args.keywords}")
    raw = pd.read_csv(args.keywords)

    # Auto-detect keyword column
    kw_col = None
    for c in raw.columns:
        if any(k in c.lower() for k in ['keyword', 'kw', 'query', 'term', 'topic']):
            kw_col = c
            break
    if kw_col is None:
        kw_col = raw.columns[0]

    # Auto-detect volume column
    vol_col = None
    for c in raw.columns:
        if any(k in c.lower() for k in ['vol', 'volume', 'search', 'msv', 'sv']):
            vol_col = c
            break

    df = pd.DataFrame({'keyword': raw[kw_col].astype(str).str.lower().str.strip()})
    if vol_col:
        df['volume'] = pd.to_numeric(raw[vol_col], errors='coerce').fillna(0)
    else:
        df['volume'] = 0

    df = df.dropna(subset=['keyword'])
    df = df[df['keyword'].str.strip() != '']
    df = df[df['keyword'] != 'nan']
    df = df.drop_duplicates('keyword').reset_index(drop=True)
    log(f"{len(df):,} unique keywords loaded")

    # ── Load embedding model (shared across all steps) ────────
    st_model = load_st_model()

    # ── Embed once — reused by all steps ─────────────────────
    embeddings = embed(df['keyword'].tolist(), st_model, batch_size=256)

    # ─────────────────────────────────────────────────────────
    # STEP 1: CLUSTER
    # ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1 / 3 — KEYWORD CLUSTERING")
    print("="*60)
    df = step_cluster(
        df, embeddings,
        threshold=args.cluster_threshold,
        sim_filter=args.sim_filter,
    )

    # ─────────────────────────────────────────────────────────
    # STEP 2: CLASSIFY
    # ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2 / 3 — NICHE + INTENT CLASSIFICATION")
    print("="*60)
    df = step_classify(
        df, embeddings, st_model,
        overlap_threshold=args.overlap_threshold,
    )

    # ─────────────────────────────────────────────────────────
    # STEP 3: TRAIN (nếu cần) + SCORE
    # ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3 / 3 — ML SCORING")
    print("="*60)

    model_path = args.model or str(out_dir / 'model.pkl')

    if args.training and not Path(model_path).exists():
        log("Training new model...")
        payload = train_model(args.training, model_path, st_model)
    elif args.training and args.retrain:
        log("Retraining model (--retrain flag)...")
        payload = train_model(args.training, model_path, st_model)
    elif Path(model_path).exists():
        log(f"Loading existing model: {model_path}")
        with open(model_path, 'rb') as f:
            payload = pickle.load(f)
        log(f"Model trained on {payload['trained_on']:,} keywords | CV AUC={payload['auc_cv']:.4f}")
    else:
        print("[ERROR] No model found. Provide --training or --model.")
        sys.exit(1)

    df = step_score(df, embeddings, payload)

    # ─────────────────────────────────────────────────────────
    # OUTPUT
    # ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("OUTPUT")
    print("="*60)

    # Column order
    cols = [
        'keyword', 'volume',
        'cluster_id', 'cluster_main', 'cluster_size', 'is_main', 'similarity',
        'primary_niche', 'secondary_niche', 'intent',
        'convert_prob', 'tier',
        'niche_score', 'intent_score',
    ]
    cols = [c for c in cols if c in df.columns]
    df_out = df[cols].sort_values(['convert_prob', 'cluster_main'], ascending=[False, True])

    # Full output
    full_path = out_dir / 'pipeline_full.csv'
    df_out.to_csv(full_path, index=False)
    log(f"Full output: {full_path} ({len(df_out):,} keywords)")

    # Tier splits
    for tier_label in ['Tier1_High', 'Tier2_Medium', 'Tier3_Low']:
        subset = df_out[df_out['tier'] == tier_label]
        if len(subset) == 0:
            continue
        tier_path = out_dir / f'pipeline_{tier_label}.csv'
        subset.to_csv(tier_path, index=False)
        arrow = ' ← làm content ngay' if 'Tier1' in tier_label else (' ← tiềm năng' if 'Tier2' in tier_label else '')
        log(f"{tier_label}: {tier_path.name} ({len(subset):,} kw){arrow}")

    # ─────────────────────────────────────────────────────────
    # SUMMARY REPORT
    # ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    tier_counts = df_out['tier'].value_counts()

    summary_lines = [
        "=" * 60,
        "PIPELINE SUMMARY",
        f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Elapsed: {elapsed:.0f}s",
        "=" * 60,
        f"\nINPUT",
        f"  Keywords file  : {args.keywords}",
        f"  Total keywords : {len(df_out):,}",
        f"\nSTEP 1 — CLUSTERING",
        f"  Threshold      : {args.cluster_threshold}",
        f"  Clusters       : {df_out['cluster_id'].nunique():,}",
        f"  Clustered kws  : {(df_out['cluster_size'] > 1).sum():,} ({(df_out['cluster_size'] > 1).mean()*100:.1f}%)",
        f"\nSTEP 2 — CLASSIFICATION",
        f"  Overlap threshold : {args.overlap_threshold}",
    ]

    for niche, cnt in df_out['primary_niche'].value_counts().items():
        summary_lines.append(f"  {niche:<20} {cnt:>5,} kw")

    summary_lines += [
        f"\nSTEP 3 — ML SCORING",
        f"  Model AUC (CV) : {payload['auc_cv']:.4f}",
        f"  Model F1  (CV) : {payload['f1_cv']:.4f}",
        f"  Trained on     : {payload['trained_on']:,} keywords",
        f"",
    ]

    print("\n── Score Distribution " + "─"*38)
    for tier in ['Tier1_High', 'Tier2_Medium', 'Tier3_Low', 'Tier4_Skip']:
        cnt = tier_counts.get(tier, 0)
        pct = cnt / len(df_out) * 100
        bar = '█' * int(pct / 2)
        arrow = ' ← làm content ngay' if 'Tier1' in tier else (' ← tiềm năng' if 'Tier2' in tier else '')
        line = f"  {tier:<18} {cnt:>5,}  ({pct:4.1f}%)  {bar}{arrow}"
        print(line)
        summary_lines.append(line)

    summary_lines += [
        f"\nTOP 20 — Highest convert probability",
    ]
    print("\n── Top 20 keywords ─────────────────────────────────")
    for _, row in df_out.head(20).iterrows():
        bar = '█' * int(row['convert_prob'] / 5)
        line = f"  {row['convert_prob']:>5.1f}%  [{row['tier'][:5]}]  {row['keyword']}  ({row['primary_niche']})"
        print(line)
        summary_lines.append(line)

    summary_txt = "\n".join(summary_lines)
    summary_path = out_dir / 'pipeline_summary.txt'
    summary_path.write_text(summary_txt, encoding='utf-8')
    log(f"Summary saved: {summary_path}")

    print(f"\n[✓] Pipeline complete in {elapsed:.0f}s → {out_dir}/")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Pinterest Affiliate Keyword Pipeline — Cluster + Classify + Score',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # ── run ──────────────────────────────────────────────────
    p_run = sub.add_parser('run', help='Full pipeline: cluster + classify + score')
    p_run.add_argument('--keywords',    required=True,  help='CSV keyword thô cần xử lý')
    p_run.add_argument('--training',    default=None,   help='CSV training data (keyword, avg_ctr, niche, intent)')
    p_run.add_argument('--model',       default=None,   help='Path model.pkl (dùng model đã train)')
    p_run.add_argument('--output',      default='output', help='Output folder (default: output/)')
    p_run.add_argument('--retrain',     action='store_true', help='Force retrain dù model.pkl đã tồn tại')
    p_run.add_argument('--cluster-threshold', type=float, default=0.82,
                       help='Clustering similarity cutoff (default: 0.82)')
    p_run.add_argument('--sim-filter',  type=float, default=0.88,
                       help='Sub-keyword similarity filter (default: 0.88)')
    p_run.add_argument('--overlap-threshold', type=float, default=0.15,
                       help='Niche overlap threshold for secondary niche (default: 0.15)')

    # ── train ─────────────────────────────────────────────────
    p_train = sub.add_parser('train', help='Chỉ train/update model')
    p_train.add_argument('--training', required=True, help='CSV training data')
    p_train.add_argument('--model',    default='model.pkl', help='Output model path')

    # ── info ──────────────────────────────────────────────────
    p_info = sub.add_parser('info', help='Xem thông tin model')
    p_info.add_argument('--model', default='model.pkl')

    args = parser.parse_args()

    if args.command == 'run':
        if not args.training and not args.model:
            parser.error("Cần ít nhất --training hoặc --model")
        run_pipeline(args)

    elif args.command == 'train':
        st_model = load_st_model()
        train_model(args.training, args.model, st_model)

    elif args.command == 'info':
        if not Path(args.model).exists():
            print(f"[ERROR] Model not found: {args.model}")
            sys.exit(1)
        with open(args.model, 'rb') as f:
            payload = pickle.load(f)
        print(f"── Model Info ──────────────────────────────────────")
        print(f"  Trained on : {payload['trained_on']:,} keywords")
        print(f"  CV AUC     : {payload['auc_cv']:.4f}")
        print(f"  CV F1      : {payload['f1_cv']:.4f}")
        print(f"  Niches     : {', '.join(payload['niches'])}")
        print(f"  Intents    : {', '.join(payload['intents'])}")


if __name__ == '__main__':
    main()
