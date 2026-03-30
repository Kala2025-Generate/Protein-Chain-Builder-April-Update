import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import io
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional

st.set_page_config(
    page_title="Protein Chain Builder",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_COMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "components", "drag_annotated_seq")
_drag_sort_component = components.declare_component("drag_annotated_seq", path=_COMP_DIR)

st.markdown("""
<style>
    .main-header { text-align: center; padding: 1rem 0; border-bottom: 2px solid #f0f0f0; margin-bottom: 2rem; }
    .status-bar {
        background: #e8f4fd; padding: 0.6rem 1rem; border-radius: 0.5rem;
        margin: 0.75rem 0; border-left: 4px solid #2196F3; font-size: 0.95rem;
    }
    .fragment-card {
        background: #ffffff; border: 1px solid #e0e0e0; border-radius: 6px;
        padding: 0.4rem 0.6rem; margin: 0.15rem 0;
    }
    .fragment-card:hover { border-color: #2196F3; box-shadow: 0 2px 6px rgba(33,150,243,0.15); }
    .construct-drop-zone {
        min-height: 160px; border: 2px dashed #bdbdbd; border-radius: 8px;
        padding: 1rem; background: #fafafa;
    }
    .sequence-mono {
        font-family: 'Courier New', monospace; background: #f5f5f5;
        padding: 0.75rem; border-radius: 4px; word-break: break-all;
        max-height: 180px; overflow-y: auto; font-size: 0.85rem;
        border: 1px solid #e0e0e0;
    }
    .stats-box { background: #e3f2fd; padding: 0.75rem 1rem; border-radius: 6px; margin-top: 0.5rem; }
    .section-divider { border-top: 1px solid #eeeeee; margin: 1rem 0; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
</style>
""", unsafe_allow_html=True)

STRICT_AA: set = set("ACDEFGHIKLMNPQRSTVWY")
LENIENT_AA: set = STRICT_AA | set("XBZUO")


# ─────────────────────────────────────────────────────────────────────────────
# Session state bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "fragments_df": None,
        "name_col": None,
        "seq_col": None,
        "selected_fragments": [],
        "current_construct_name": "Construct_1",
        "saved_constructs": [],
        "search_term": "",
        "validation_strict": True,
        "rejected_rows": None,
        "drag_reorder_count": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers  (no Streamlit calls – safe to cache)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    name_patterns = ["fragment_name", "name", "protein_name", "id", "label"]
    seq_patterns = ["sequence", "aa_sequence", "aa", "seq"]
    name_candidates, seq_candidates = [], []
    for col in df.columns:
        cl = col.lower().strip()
        if any(p in cl for p in name_patterns):
            name_candidates.append(col)
        if any(p in cl for p in seq_patterns):
            seq_candidates.append(col)
    return {"name": name_candidates, "sequence": seq_candidates}


def _validate_sequence(sequence: str, strict: bool = True) -> bool:
    if not sequence or not isinstance(sequence, str):
        return False
    valid = STRICT_AA if strict else LENIENT_AA
    return all(c in valid for c in sequence.strip().upper())


def _clean_and_validate(
    df: pd.DataFrame, name_col: str, seq_col: str, strict: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clean_df = df.copy()
    clean_df[name_col] = clean_df[name_col].astype(str).str.strip()
    clean_df[seq_col] = clean_df[seq_col].astype(str).str.strip()

    rejected: List[dict] = []
    drop_idx: List[int] = []

    for idx, row in clean_df.iterrows():
        name = row[name_col]
        seq = row[seq_col]
        reason = None

        if not name or name in ("", "nan"):
            reason = "Empty name"
        elif not seq or seq in ("", "nan"):
            reason = "Empty sequence"
        elif not _validate_sequence(seq, strict):
            reason = f"Invalid amino acids (strict={strict})"

        if reason:
            rejected.append({"original_index": idx, "name": name, "sequence": seq, "reason": reason})
            drop_idx.append(idx)

    clean_df = clean_df.drop(drop_idx)
    rejected_df = pd.DataFrame(rejected) if rejected else pd.DataFrame()
    return clean_df, rejected_df


def _sequence_stats(sequence: str) -> Dict:
    if not sequence:
        return {}
    composition = {
        aa: round(sequence.count(aa) / len(sequence) * 100, 1)
        for aa in STRICT_AA
        if sequence.count(aa) > 0
    }
    return {"length": len(sequence), "composition": composition}


def _estimate_pi(sequence: str) -> float:
    if not sequence:
        return 0.0
    pos = sequence.count("K") + sequence.count("R") + sequence.count("H")
    neg = sequence.count("D") + sequence.count("E")
    if pos + neg == 0:
        return 7.0
    return round(7.0 + (pos - neg) / len(sequence) * 5, 2)


def _filter_fragments(
    df: pd.DataFrame,
    name_col: str,
    seq_col: str,
    search: str,
    min_len: int,
    max_len: int,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if search:
        s = search.lower()
        mask = out[name_col].str.lower().str.contains(s, na=False) | \
               out[seq_col].str.lower().str.contains(s, na=False)
        out = out[mask]
    out = out[out[seq_col].str.len().between(min_len, max_len)]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Render helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_header() -> None:
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🧬 Protein Chain Builder")
    st.markdown("Build protein constructs by assembling fragment sequences into chains.")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📖 Quick Start Guide", expanded=False):
        st.markdown("""
        1. **Upload** a CSV or Excel file containing fragment names and amino acid sequences.
        2. **Map columns** – select which columns hold names and sequences.
        3. **Process** the file; the app validates and loads your fragments.
        4. **Add fragments** from the library into the Construct Builder.
        5. **Reorder** fragments with the ⬆️ / ⬇️ buttons.
        6. **Save** each construct and **Export** everything to CSV when done.
        """)


def render_file_upload() -> bool:
    """Returns True once fragments are loaded into session state."""
    st.header("📁 Upload Fragment Library")

    uploaded = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="File must contain a fragment-name column and an amino-acid sequence column.",
    )

    if uploaded is None:
        if st.session_state.fragments_df is not None:
            st.info("✅ Fragment library already loaded – scroll down to continue building.")
            return True
        return False

    # Read bytes once
    file_bytes = uploaded.read()

    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        return False

    if df.empty:
        st.error("The uploaded file is empty.")
        return False

    candidates = _detect_columns(df)
    all_cols = df.columns.tolist()

    col_a, col_b = st.columns(2)
    with col_a:
        default_name = candidates["name"][0] if candidates["name"] else all_cols[0]
        name_col = st.selectbox(
            "Fragment Name Column",
            all_cols,
            index=all_cols.index(default_name),
            help="Column containing fragment identifiers.",
        )
    with col_b:
        default_seq = candidates["sequence"][0] if candidates["sequence"] else all_cols[min(1, len(all_cols) - 1)]
        seq_col = st.selectbox(
            "Sequence Column",
            all_cols,
            index=all_cols.index(default_seq),
            help="Column containing amino acid sequences.",
        )

    strict = st.checkbox(
        "Strict validation (standard 20 amino acids only)",
        value=st.session_state.validation_strict,
        help="Uncheck to allow ambiguous codes: X, B, Z, U, O",
    )
    st.session_state.validation_strict = strict

    if st.button("Process File", type="primary"):
        with st.spinner("Validating and loading fragments…"):
            clean_df, rejected_df = _clean_and_validate(df, name_col, seq_col, strict)

        if clean_df.empty:
            st.error("❌ No valid fragments found. Check column selection or validation settings.")
            if not rejected_df.empty:
                st.dataframe(rejected_df, use_container_width=True)
            return False

        # Persist to session state
        st.session_state.fragments_df = clean_df.reset_index(drop=True)
        st.session_state.name_col = name_col
        st.session_state.seq_col = seq_col
        st.session_state.rejected_rows = rejected_df

        st.success(f"✅ Loaded **{len(clean_df)}** valid fragments.")

        # Summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Fragments", len(clean_df))
        m2.metric("Unique Sequences", clean_df[seq_col].nunique())
        rejected_count = len(rejected_df) if not rejected_df.empty else 0
        m3.metric("Rejected Rows", rejected_count)

        # Preview
        with st.expander("📊 Data Preview (first 5 rows)", expanded=True):
            preview = clean_df[[name_col, seq_col]].head().copy()
            preview["Length"] = preview[seq_col].str.len()
            st.dataframe(preview, use_container_width=True)

        # Rejected rows
        if not rejected_df.empty:
            with st.expander(f"⚠️ {len(rejected_df)} Rejected Rows", expanded=False):
                st.dataframe(rejected_df, use_container_width=True)
                st.download_button(
                    "📥 Download Rejected Rows CSV",
                    data=rejected_df.to_csv(index=False),
                    file_name=f"rejected_{datetime.now():%Y%m%d_%H%M%S}.csv",
                    mime="text/csv",
                )

        st.rerun()

    return st.session_state.fragments_df is not None


def render_fragment_library() -> None:
    df: pd.DataFrame = st.session_state.fragments_df
    name_col: str = st.session_state.name_col
    seq_col: str = st.session_state.seq_col

    st.header("🧪 Fragment Library")

    search = st.text_input(
        "🔍 Search",
        value=st.session_state.search_term,
        placeholder="Name or sequence…",
        key="search_input",
    )
    st.session_state.search_term = search

    c1, c2 = st.columns(2)
    min_len = c1.number_input("Min Length", min_value=0, value=0, step=1)
    max_len = c2.number_input("Max Length", min_value=1, value=10_000, step=1)

    filtered = _filter_fragments(df, name_col, seq_col, search, min_len, max_len)
    st.caption(f"Showing **{len(filtered)}** of **{len(df)}** fragments")

    if filtered.empty:
        st.info("No fragments match the current filters.")
        return

    # Pagination
    per_page = 20
    total_pages = max(1, (len(filtered) - 1) // per_page + 1)
    page = 0
    if total_pages > 1:
        page = st.selectbox("Page", range(1, total_pages + 1), key="lib_page") - 1

    page_df = filtered.iloc[page * per_page : (page + 1) * per_page]

    for idx, row in page_df.iterrows():
        name = row[name_col]
        seq: str = str(row[seq_col])
        length = len(seq)
        preview = seq[:22] + "…" if length > 22 else seq

        c_info, c_btn = st.columns([4, 1])
        with c_info:
            st.markdown(
                f'<div class="fragment-card"><strong>{name}</strong>'
                f'<br><span style="color:#888;font-size:0.8rem">{length} aa &nbsp;·&nbsp; {preview}</span></div>',
                unsafe_allow_html=True,
            )
        with c_btn:
            if st.button("➕", key=f"add_{idx}", help="Add to builder"):
                st.session_state.selected_fragments.append(
                    {"name": name, "sequence": seq, "length": length}
                )
                st.rerun()


def render_construct_builder() -> None:
    st.header("🔗 Construct Builder")

    st.session_state.current_construct_name = st.text_input(
        "Construct Name",
        value=st.session_state.current_construct_name,
        key="construct_name_input",
    )

    frags: List[dict] = st.session_state.selected_fragments

    if not frags:
        st.info("Add fragments from the library on the left.")
    else:
        st.caption(f"**{len(frags)} fragment(s)** in this construct")
        for i, frag in enumerate(frags):
            c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
            c1.markdown(f"**{frag['name']}** &nbsp; `{frag['length']} aa`", unsafe_allow_html=True)

            if c2.button("⬆️", key=f"up_{i}", disabled=(i == 0), help="Move up"):
                frags[i], frags[i - 1] = frags[i - 1], frags[i]
                st.rerun()

            if c3.button("⬇️", key=f"dn_{i}", disabled=(i == len(frags) - 1), help="Move down"):
                frags[i], frags[i + 1] = frags[i + 1], frags[i]
                st.rerun()

            if c4.button("🗑️", key=f"rm_{i}", help="Remove"):
                frags.pop(i)
                st.rerun()

    # ── Live concatenated sequence preview ───────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    if frags:
        cat_seq = "".join(f["sequence"] for f in frags)
        st.markdown(
            f'<div class="status-bar">⛓️ <strong>{len(frags)} fragment(s) &nbsp;·&nbsp; '
            f'{len(cat_seq)} aa total</strong></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "**🧬 Annotated Concatenated Sequence** "
            "<span style='font-size:0.75rem;color:#888;font-weight:normal'>"
            "— drag coloured blocks to reorder, or use ⬆️ ⬇️ buttons above</span>",
            unsafe_allow_html=True,
        )

        PALETTE = [
            ("#1565C0", "#E3F2FD"),  # blue
            ("#B71C1C", "#FFEBEE"),  # red
            ("#1B5E20", "#E8F5E9"),  # green
            ("#E65100", "#FFF3E0"),  # orange
            ("#4A148C", "#F3E5F5"),  # purple
            ("#006064", "#E0F7FA"),  # teal
            ("#F57F17", "#FFFDE7"),  # yellow
            ("#37474F", "#ECEFF1"),  # grey
        ]

        drag_frags = [{"name": f["name"], "sequence": f["sequence"]} for f in frags]
        new_order = _drag_sort_component(
            fragments=drag_frags,
            palette=[list(p) for p in PALETTE],
            key=f"drag_annotated_{st.session_state.drag_reorder_count}",
        )
        if new_order is not None and list(new_order) != list(range(len(frags))):
            st.session_state.selected_fragments = [frags[i] for i in new_order]
            st.session_state.drag_reorder_count += 1
            st.rerun()

        # Plain copyable version
        with st.expander("📋 Copy plain sequence", expanded=False):
            st.code(cat_seq, language=None)

        if st.checkbox("Show composition analysis", key="comp_analysis"):
            stats = _sequence_stats(cat_seq)
            pi_val = _estimate_pi(cat_seq)
            st.markdown('<div class="stats-box">', unsafe_allow_html=True)
            st.write(f"**Length:** {stats['length']} aa &nbsp;|&nbsp; **Estimated pI:** {pi_val}")
            if "composition" in stats:
                comp_str = " · ".join(
                    f"{aa}: {pct}%" for aa, pct in sorted(stats["composition"].items())
                )
                st.caption(comp_str)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)

    if b1.button("🗑️ Clear All", use_container_width=True):
        st.session_state.selected_fragments = []
        st.rerun()

    if b2.button("💾 Save Construct", disabled=not frags, type="primary", use_container_width=True):
        _save_construct()
        st.success(f"Saved **{st.session_state.current_construct_name}**!")
        st.rerun()

    if b3.button("🆕 New Construct", use_container_width=True):
        st.session_state.selected_fragments = []
        n = len(st.session_state.saved_constructs) + 1
        st.session_state.current_construct_name = f"Construct_{n}"
        st.rerun()


def _save_construct() -> None:
    frags = st.session_state.selected_fragments
    if not frags:
        return
    seq = "".join(f["sequence"] for f in frags)
    st.session_state.saved_constructs.append(
        {
            "name": st.session_state.current_construct_name,
            "fragments": frags.copy(),
            "sequence": seq,
            "length": len(seq),
            "fragment_count": len(frags),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )


def render_output_panel() -> None:
    st.header("📊 Output & Export")

    # ── Saved constructs ──────────────────────────────────────────────────────
    saved: List[dict] = st.session_state.saved_constructs
    if saved:
        st.subheader(f"💾 Saved Constructs ({len(saved)})")
        for i, c in enumerate(saved):
            with st.expander(f"{c['name']}  —  {c['length']} aa, {c['fragment_count']} fragment(s)"):
                st.caption(f"Created: {c['created_at']}")
                st.write(" → ".join(f["name"] for f in c["fragments"]))
                st.code(c["sequence"], language=None)
                if st.button("🗑️ Remove", key=f"del_c_{i}"):
                    saved.pop(i)
                    st.rerun()

    # ── Export ────────────────────────────────────────────────────────────────
    st.subheader("📥 Export")
    frags: List[dict] = st.session_state.selected_fragments
    inc_current = st.checkbox("Include current (unsaved) construct", value=True)
    inc_components = st.checkbox("Also export component mapping CSV", value=False)

    has_data = bool(saved) or (inc_current and bool(frags))
    if st.button("📊 Generate Export", type="primary", disabled=not has_data):
        _do_export(inc_current, inc_components)


def _do_export(include_current: bool, include_components: bool) -> None:
    frags = st.session_state.selected_fragments
    to_export: List[dict] = list(st.session_state.saved_constructs)

    if include_current and frags:
        seq = "".join(f["sequence"] for f in frags)
        to_export.append(
            {
                "name": st.session_state.current_construct_name,
                "fragments": frags.copy(),
                "sequence": seq,
                "length": len(seq),
                "fragment_count": len(frags),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    if not to_export:
        st.warning("Nothing to export.")
        return

    rows = [
        {
            "construct_name": c["name"],
            "concatenated_sequence": c["sequence"],
            "length": c["length"],
            "fragment_count": c["fragment_count"],
            "fragment_names": " | ".join(f["name"] for f in c["fragments"]),
            "created_at": c["created_at"],
        }
        for c in to_export
    ]
    csv_main = pd.DataFrame(rows).to_csv(index=False)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "📥 Download Constructs CSV",
        data=csv_main,
        file_name=f"protein_constructs_{ts}.csv",
        mime="text/csv",
        key="dl_constructs",
    )

    if include_components:
        comp_rows = [
            {
                "construct_name": c["name"],
                "position": i,
                "fragment_name": f["name"],
                "fragment_sequence": f["sequence"],
                "fragment_length": f["length"],
            }
            for c in to_export
            for i, f in enumerate(c["fragments"])
        ]
        csv_comp = pd.DataFrame(comp_rows).to_csv(index=False)
        st.download_button(
            "📥 Download Component Mapping CSV",
            data=csv_comp,
            file_name=f"protein_components_{ts}.csv",
            mime="text/csv",
            key="dl_components",
        )

    st.success(f"✅ Ready to download — {len(to_export)} construct(s) exported.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()
    render_header()

    file_ready = render_file_upload()

    if file_ready:
        st.divider()
        col_lib, col_build, col_out = st.columns([1, 1, 1])
        with col_lib:
            render_fragment_library()
        with col_build:
            render_construct_builder()
        with col_out:
            render_output_panel()

    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#9e9e9e;font-size:0.85rem;padding:0.5rem'>"
        "🧬 Protein Chain Builder &nbsp;·&nbsp; Built with Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
