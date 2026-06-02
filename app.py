import streamlit as st
import pandas as pd
import json
import html
import re
from datetime import date
from groq import Groq
import io

st.set_page_config(
    page_title="ChurnGuard — Retention Agent",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0f1117; }
    [data-testid="stSidebar"] { background: #13151f; border-right: 1px solid #2a2d3a; }

    .hero { background: linear-gradient(135deg, #1a1d27 0%, #0f1117 100%);
            border: 1px solid #2a2d3a; border-radius: 16px; padding: 36px 40px; margin-bottom: 28px; }
    .hero h1 { font-size: 2.2rem; font-weight: 800; color: #ffffff; margin: 0 0 8px 0; }
    .hero p  { color: #9ca3b0; font-size: 1.05rem; margin: 0; line-height: 1.6; }

    .stat-card { background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 12px;
                 padding: 20px 24px; text-align: center; margin-bottom: 8px; }
    .stat-num  { font-size: 2rem; font-weight: 800; color: #4a9eff; }
    .stat-lbl  { font-size: 0.78rem; color: #6b7280; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }

    .email-card { background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 14px;
                  padding: 28px; margin-bottom: 20px; }
    .email-header { display:flex; justify-content:space-between; align-items:flex-start;
                    margin-bottom: 18px; flex-wrap: wrap; gap: 8px; }
    .customer-name { font-size: 1.05rem; font-weight: 700; color: #f0f2f5; }
    .customer-email { color: #6b7280; font-size: 0.85rem; margin-top: 2px; }
    .customer-meta  { color: #6b7280; font-size: 0.8rem; margin-top: 4px; }

    .badge-high   { background:#ff4b4b22; border:1px solid #ff4b4b; border-radius:20px;
                    padding:3px 12px; color:#ff4b4b; font-size:0.78rem; font-weight:700; white-space:nowrap; }
    .badge-medium { background:#ffa50022; border:1px solid #ffa500; border-radius:20px;
                    padding:3px 12px; color:#ffa500; font-size:0.78rem; font-weight:700; white-space:nowrap; }

    .reasoning { background: #0d1117; border-left: 3px solid #4a9eff; border-radius: 0 8px 8px 0;
                 padding: 12px 16px; color: #7a8599; font-size: 0.83rem; margin-bottom: 18px; line-height: 1.65; }

    .step-box { background:#1a1d27; border:1px solid #2a2d3a; border-radius:12px;
                padding:18px 22px; margin-bottom:10px; }
    .step-num { color:#4a9eff; font-weight:800; font-size:1rem; }

    .footer { text-align:center; color:#4a5568; font-size:0.78rem; margin-top:48px;
              padding-top:20px; border-top:1px solid #2a2d3a; }

    h2, h3 { color: #f0f2f5 !important; }
    .stButton>button { border-radius:8px; font-weight:600; }
    .stDownloadButton>button { border-radius:8px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


SAMPLE_CSV = """customer_name,email,last_purchase_date,total_orders,total_spend,product_category
Sarah Mitchell,sarah@example.com,2026-05-20,12,1840.00,Skincare
Elena Vasquez,elena@example.com,2026-05-30,5,350.00,Skincare
Priya Nair,priya@example.com,2026-04-10,3,280.00,Fitness
Marcus Webb,marcus@example.com,2025-08-01,3,1350.00,Skincare
Ryan Kowalski,ryan@example.com,2025-07-15,3,1100.00,Supplements
Fatima Al-Hassan,fatima@example.com,2026-03-10,6,820.00,Fitness
James Okafor,james@example.com,2026-03-01,4,380.00,Supplements
David Chen,david@example.com,2026-03-18,5,430.00,Fitness
Tom Bergmann,tom@example.com,2025-08-10,2,95.00,Supplements
Aisha Patel,aisha@example.com,2025-07-01,1,45.00,Supplements
"""


def score_customers(df: pd.DataFrame) -> pd.DataFrame:
    today = date.today()
    df = df.copy()
    df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"], errors="coerce")
    df = df.dropna(subset=["last_purchase_date"])
    df["days_since_purchase"] = df["last_purchase_date"].apply(lambda d: (today - d.date()).days)

    df["r_score"] = pd.cut(df["days_since_purchase"],
        bins=[0, 30, 60, 90, 180, 99999], labels=[5, 4, 3, 2, 1], right=True).astype(float)
    df["f_score"] = pd.cut(df["total_orders"],
        bins=[0, 1, 3, 6, 10, 99999], labels=[1, 2, 3, 4, 5], right=True).astype(float)
    df["m_score"] = pd.cut(df["total_spend"],
        bins=[0, 100, 300, 700, 1200, 99999], labels=[1, 2, 3, 4, 5], right=True).astype(float)

    df["rfm_score"] = df["r_score"] * 0.5 + df["f_score"] * 0.3 + df["m_score"] * 0.2
    df["churn_risk_score"] = ((5 - df["rfm_score"]) / 4 * 100).round(1)
    df["risk_level"] = df["churn_risk_score"].apply(
        lambda s: "High" if s >= 65 else ("Medium" if s >= 35 else "Low"))
    df["ltv_tier"] = df["total_spend"].apply(
        lambda x: "High" if x >= 500 else ("Medium" if x >= 150 else "Low"))
    return df.sort_values("churn_risk_score", ascending=False)


def generate_retention_email(client: Groq, customer: dict, model: str) -> dict:
    system_prompt = """You are a senior CRM manager at a premium subscription brand writing a re-engagement email to a lapsed high-value customer.

Rules:
- Professional, warm tone — like a trusted brand, not a salesperson
- Open with ONLY the customer's first name followed by a comma, then a blank line — no "Dear", no "Hi", no "Hello"
- Do NOT open with "I hope", "We noticed", "It's been a while", "We miss you", or "Just checking in"
- Leave a blank line before the sign-off, like a proper email
- Paragraph 1: lead with something genuinely useful — a seasonal insight, new product development, or expert tip relevant to their specific category
- Paragraph 2: a personalised recommendation or next step based on their category and purchase history
- Paragraph 3: a soft, clear call to action — no pressure, no discount, no urgency language
- Sign off warmly and close as "The [actual product category] Team"
- Professional tone, no exclamation marks, no emojis, no markdown
- 120 to 160 words total
- Subject line: specific and curiosity-driven, avoid "We miss you", "Check this out", "Don't miss"

Respond in JSON with exactly these three keys:
{
  "reasoning": "2-3 sentences on why this customer is at risk and what angle makes sense for them",
  "subject": "email subject line",
  "body": "full professional email body with natural line breaks"
}"""

    user_prompt = f"""Customer:
- Name: {customer['customer_name']}
- Category: {customer['product_category']}
- Days since last purchase: {customer['days_since_purchase']}
- Total orders: {int(customer['total_orders'])}
- Total spend: ${float(customer['total_spend']):.2f}
- Churn risk: {customer['churn_risk_score']}%
- LTV tier: {customer['ltv_tier']}"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=0.7,
        max_tokens=600,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    # Replace curly/smart quotes with straight quotes
    raw = raw.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            reasoning = re.search(r'"reasoning"\s*:\s*"(.*?)"(?=\s*,\s*"subject")', raw, re.DOTALL)
            subject   = re.search(r'"subject"\s*:\s*"(.*?)"(?=\s*,\s*"body")', raw, re.DOTALL)
            body      = re.search(r'"body"\s*:\s*"(.*?)"(?=\s*\})', raw, re.DOTALL)
            if reasoning and subject and body:
                return {
                    "reasoning": reasoning.group(1).replace("\\n", "\n"),
                    "subject":   subject.group(1),
                    "body":      body.group(1).replace("\\n", "\n"),
                }
        except Exception:
            pass
        # Last resort — return raw body so nothing is lost
        return {
            "reasoning": "AI responded but formatting was unexpected.",
            "subject": "A quick note from us",
            "body": raw
        }


def main():

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
<div style="font-size:0.9rem; font-weight:700; color:#9ca3b0; text-transform:uppercase;
            letter-spacing:0.08em; margin-bottom:16px;">⚙️ Settings</div>
""", unsafe_allow_html=True)

        # ── API Key ───────────────────────────────────────────────────────────
        try:
            secret_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            secret_key = ""

        if secret_key:
            groq_key = st.session_state.get("override_key", secret_key)
            st.markdown("""
<div style="background:#00c85318; border:1px solid #00c853; border-radius:8px;
            padding:10px 14px; font-size:0.85rem; color:#00c853; margin-bottom:10px;">
✓ &nbsp; Groq API key configured
</div>""", unsafe_allow_html=True)
            if st.session_state.get("override_key"):
                st.markdown("""
<div style="background:#00c85318; border:1px solid #00c853; border-radius:8px;
            padding:10px 14px; font-size:0.85rem; color:#00c853; margin-bottom:6px;">
✓ &nbsp; Override key saved for this session
</div>""", unsafe_allow_html=True)
                if st.button("Remove override key", use_container_width=True):
                    st.session_state.pop("override_key", None)
                    st.rerun()
            else:
                override_input = st.text_input(
                    "Override API Key",
                    type="password",
                    placeholder="Paste your key here — starts with gsk_",
                    help="Replaces the configured key for this session"
                )
                if st.button("Save Override Key", use_container_width=True, type="primary"):
                    if override_input:
                        st.session_state["override_key"] = override_input
                        st.rerun()
                    else:
                        st.warning("Paste your API key first.")
        else:
            saved_key = st.session_state.get("groq_key", "")
            if saved_key:
                groq_key = saved_key
                st.markdown("""
<div style="background:#00c85318; border:1px solid #00c853; border-radius:8px;
            padding:10px 14px; font-size:0.85rem; color:#00c853; margin-bottom:6px;">
✓ &nbsp; API key saved for this session
</div>""", unsafe_allow_html=True)
                if st.button("Remove API key", use_container_width=True):
                    st.session_state.pop("groq_key", None)
                    st.rerun()
            else:
                groq_key = ""
                typed_key = st.text_input(
                    "Groq API Key",
                    type="password",
                    placeholder="Paste your key here — starts with gsk_",
                    help="Free at console.groq.com — no credit card needed",
                    key="groq_input"
                )
                if st.button("Save API Key", use_container_width=True, type="primary"):
                    if typed_key:
                        st.session_state["groq_key"] = typed_key
                        groq_key = typed_key
                        st.rerun()
                    else:
                        st.warning("Paste your API key first.")
                st.markdown("""
<div style="background:#0d1117; border:1px solid #2a2d3a; border-radius:8px;
            padding:12px 14px; font-size:0.8rem; color:#9ca3b0; line-height:1.8; margin-top:4px;">
<strong style="color:#f0f2f5;">Get a free key in 2 minutes</strong><br>
1. <a href="https://console.groq.com" target="_blank" style="color:#4a9eff;">console.groq.com</a> → Sign up<br>
2. API Keys → Create key<br>
3. Copy and paste it above
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div style="background:#0d1117; border:1px solid #2a2d3a; border-radius:8px;
            padding:12px 14px; font-size:0.8rem; color:#9ca3b0; line-height:1.6; margin-top:10px;">
<strong style="color:#f0f2f5;">Why Groq, not ChatGPT or Claude?</strong><br>
Both require a paid API plan. Groq is free, same quality (Llama 3.3 70B), and faster.
</div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── CSV format ────────────────────────────────────────────────────────
        st.markdown("""
<div style="font-size:0.82rem; font-weight:600; color:#f0f2f5; margin-bottom:8px;">CSV format required</div>
<div style="display:flex; flex-direction:column; gap:4px;">
  <div style="background:#0d1117; border:1px solid #2a2d3a; border-radius:6px; padding:5px 10px; font-size:0.75rem; color:#9ca3b0; font-family:monospace;">customer_name</div>
  <div style="background:#0d1117; border:1px solid #2a2d3a; border-radius:6px; padding:5px 10px; font-size:0.75rem; color:#9ca3b0; font-family:monospace;">email</div>
  <div style="background:#0d1117; border:1px solid #2a2d3a; border-radius:6px; padding:5px 10px; font-size:0.75rem; color:#9ca3b0; font-family:monospace;">last_purchase_date &nbsp;<span style="color:#4a5568;">YYYY-MM-DD</span></div>
  <div style="background:#0d1117; border:1px solid #2a2d3a; border-radius:6px; padding:5px 10px; font-size:0.75rem; color:#9ca3b0; font-family:monospace;">total_orders</div>
  <div style="background:#0d1117; border:1px solid #2a2d3a; border-radius:6px; padding:5px 10px; font-size:0.75rem; color:#9ca3b0; font-family:monospace;">total_spend</div>
  <div style="background:#0d1117; border:1px solid #2a2d3a; border-radius:6px; padding:5px 10px; font-size:0.75rem; color:#9ca3b0; font-family:monospace;">product_category</div>
</div>""", unsafe_allow_html=True)
        st.markdown("")
        st.download_button("📥 Download sample CSV", data=SAMPLE_CSV,
                           file_name="sample_customers.csv", mime="text/csv",
                           use_container_width=True)

        st.markdown("---")

        # ── Advanced ──────────────────────────────────────────────────────────
        with st.expander("Advanced — change AI model"):
            selected_model = st.selectbox("Model", [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768"
            ], index=0)
            st.session_state["model"] = selected_model

    model = st.session_state.get("model", "llama-3.3-70b-versatile")

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="hero">
  <h1>🛡️ ChurnGuard</h1>
  <p>Upload your customer list. ChurnGuard scores every customer for churn risk, identifies the ones
  worth spending retention budget on, and writes a personalised re-engagement email for each —
  in under 60 seconds. Built for subscription businesses. No engineering required.</p>
</div>
""", unsafe_allow_html=True)

    # ── Upload ────────────────────────────────────────────────────────────────
    col_up, col_sample, col_reset = st.columns([3, 1, 1])
    with col_up:
        uploaded = st.file_uploader("Upload your customer CSV", type=["csv"], label_visibility="collapsed")
    with col_sample:
        use_sample = st.button("▶ Try sample data", use_container_width=True, type="secondary")
    with col_reset:
        if st.button("↺ Reset", use_container_width=True):
            for k in ["df_raw", "email_results"]:
                st.session_state.pop(k, None)
            st.rerun()

    if uploaded:
        st.session_state["df_raw"] = pd.read_csv(uploaded)
        st.session_state.pop("email_results", None)
        st.success(f"Loaded {len(st.session_state['df_raw'])} customers.")
    elif use_sample:
        st.session_state["df_raw"] = pd.read_csv(io.StringIO(SAMPLE_CSV))
        st.session_state.pop("email_results", None)
        st.info("Using sample data — 10 customers across Skincare, Fitness, and Supplements.")

    df_raw = st.session_state.get("df_raw", None)

    # ── Landing ───────────────────────────────────────────────────────────────
    if df_raw is None:
        st.markdown("---")
        st.markdown("### How it works")
        for num, title, desc in [
            ("1", "Upload your customer list",
             "Use the uploader above, or click <strong>Try sample data</strong> to see it running immediately. No API key needed for the scoring step."),
            ("2", "Every customer is scored for churn risk",
             "ChurnGuard runs a weighted RFM model — Recency (50%), Frequency (30%), Monetary value (20%) — and gives each customer a churn risk score from 0–100%."),
            ("3", "High-value at-risk customers are surfaced",
             "Only customers with <strong>high lifetime value AND high churn risk</strong> are flagged for outreach. Emailing everyone wastes budget and trains low-value customers to expect discounts."),
            ("4", "AI writes a personalised email for each one",
             "One click. The agent reasons about each customer individually and writes a specific, value-led email — not a template. You see the reasoning so you can review before sending."),
            ("5", "Export and send",
             "Download all emails as CSV and paste into Gmail, Outlook, or Mailchimp — whatever your team already uses."),
        ]:
            st.markdown(f"""
<div class="step-box">
<span class="step-num">Step {num} —</span> <strong>{title}</strong><br>
<span style="color:#9ca3b0; font-size:0.88rem; line-height:1.6;">{desc}</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="footer">
ChurnGuard · Built in 3 hours, AI-assisted · Groq + Llama 3.3 70B ·
Deploy free at <a href="https://share.streamlit.io" style="color:#4a9eff;">share.streamlit.io</a>
</div>
""", unsafe_allow_html=True)
        return

    # ── Validate ──────────────────────────────────────────────────────────────
    required_cols = {"customer_name", "email", "last_purchase_date", "total_orders", "total_spend", "product_category"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        st.error(f"Your CSV is missing these columns: {', '.join(sorted(missing))}")
        st.caption("Download the sample CSV from the sidebar to see the exact format needed.")
        return

    for num_col in ["total_orders", "total_spend"]:
        df_raw[num_col] = pd.to_numeric(df_raw[num_col], errors="coerce")
    if df_raw[["total_orders", "total_spend"]].isnull().any().any():
        st.error("Some rows have non-numeric values in total_orders or total_spend. Please fix your CSV and re-upload.")
        return

    df = score_customers(df_raw)

    # ── Overview metrics ──────────────────────────────────────────────────────
    st.markdown("---")
    total = len(df)
    high  = len(df[df["risk_level"] == "High"])
    med   = len(df[df["risk_level"] == "Medium"])
    act   = len(df[(df["risk_level"].isin(["High","Medium"])) & (df["ltv_tier"] == "High")])

    c1, c2, c3, c4 = st.columns(4)
    for col, num, lbl in [(c1, total, "Total Customers"), (c2, high, "High Risk"),
                          (c3, med, "Medium Risk"),       (c4, act, "High-Value At Risk")]:
        col.markdown(f"""
<div class="stat-card"><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div>
""", unsafe_allow_html=True)

    # ── Risk table ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Customer Risk Scores")

    f1, f2 = st.columns(2)
    with f1:
        risk_filter = st.multiselect("Filter by risk", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    with f2:
        ltv_filter = st.multiselect("Filter by LTV tier", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])

    df_view = df[df["risk_level"].isin(risk_filter) & df["ltv_tier"].isin(ltv_filter)]
    display_cols = ["customer_name", "email", "product_category", "days_since_purchase",
                    "total_orders", "total_spend", "churn_risk_score", "risk_level", "ltv_tier"]

    st.dataframe(
        df_view[display_cols].rename(columns={
            "customer_name": "Name", "email": "Email", "product_category": "Category",
            "days_since_purchase": "Days Since Purchase", "total_orders": "Orders",
            "total_spend": "Spend ($)", "churn_risk_score": "Churn Risk %",
            "risk_level": "Risk", "ltv_tier": "LTV Tier"
        }),
        use_container_width=True, hide_index=True,
        column_config={
            "Churn Risk %": st.column_config.ProgressColumn(
                "Churn Risk %", min_value=0, max_value=100, format="%.1f%%"),
            "Spend ($)": st.column_config.NumberColumn("Spend ($)", format="$%.2f"),
        }
    )

    st.download_button("📥 Export scored data (CSV)",
                       data=df_view[display_cols].to_csv(index=False),
                       file_name="churnguard_scored.csv", mime="text/csv")

    # ── Email generation ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Generate Retention Emails")

    targets = df[(df["risk_level"].isin(["High", "Medium"])) & (df["ltv_tier"] == "High")]

    if len(targets) == 0:
        st.info("No high-value at-risk customers found in your data.")
        return

    st.markdown(f"**{len(targets)} customers** selected for retention outreach.")
    st.caption("Emails are generated for High and Medium risk customers with High LTV only — regardless of the table filter above. This is intentional: emailing low-value customers trains them to wait for discounts.")

    if not groq_key and "email_results" not in st.session_state:
        st.warning("Add your free Groq API key in the sidebar to generate emails. Step-by-step instructions are there.")
        return

    if "email_results" not in st.session_state:
        if st.button(f"🤖 Generate emails for {len(targets)} customers", type="primary", use_container_width=True):
            client = Groq(api_key=groq_key)
            results = []
            fatal_error = None
            bar = st.progress(0, text="Agent starting...")
            for i, (_, row) in enumerate(targets.iterrows()):
                bar.progress((i) / len(targets), text=f"Writing email for {row['customer_name']} ({i+1}/{len(targets)})...")
                try:
                    out = generate_retention_email(client, row.to_dict(), model)
                    results.append({**row.to_dict(), **out})
                except Exception as e:
                    err = str(e).lower()
                    if "invalid_api_key" in err or "authentication" in err or "401" in err:
                        fatal_error = "Invalid API key. Please check your Groq API key in the sidebar and try again."
                    elif "rate_limit" in err or "429" in err or "too many" in err:
                        fatal_error = "Groq rate limit reached. Wait a minute and try again, or switch to a different model in Advanced settings."
                    elif "token" in err and "limit" in err:
                        fatal_error = "Token limit exceeded. Try switching to llama-3.1-8b-instant in Advanced settings."
                    elif "connection" in err or "timeout" in err:
                        fatal_error = "Connection error. Check your internet and try again."
                    else:
                        fatal_error = "Something went wrong generating emails. Please try again."
                    break

            if fatal_error:
                bar.empty()
                st.error(fatal_error)
            else:
                bar.progress(1.0, text=f"Done — {len(results)} emails ready.")
                st.session_state["email_results"] = results
                st.rerun()
        return

    # ── Show emails ───────────────────────────────────────────────────────────
    results = st.session_state["email_results"]

    col_hdr, col_regen = st.columns([3, 1])
    with col_hdr:
        st.markdown(f"### {len(results)} Emails Ready")
    with col_regen:
        if st.button("↺ Regenerate", use_container_width=True):
            st.session_state.pop("email_results", None)
            st.rerun()

    export_rows = []
    for i, r in enumerate(results):
        badge_cls = "badge-high" if r["risk_level"] == "High" else "badge-medium"
        safe_reasoning = html.escape(r["reasoning"])

        # Customer header card
        st.markdown(f"""
<div class="email-card">
  <div class="email-header">
    <div>
      <div class="customer-name">{html.escape(r['customer_name'])}</div>
      <div class="customer-email">{html.escape(r['email'])}</div>
      <div class="customer-meta">{r['days_since_purchase']} days since last purchase · ${float(r['total_spend']):.0f} lifetime spend · {r['product_category']}</div>
    </div>
    <span class="{badge_cls}">{r['risk_level']} Risk</span>
  </div>
  <div class="reasoning">🧠 <strong>Why this email:</strong> {safe_reasoning}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("**Subject**")
        st.code(r["subject"], language=None)

        st.markdown("**Email body**")
        st.code(r["body"], language=None)

        st.markdown("---")

        export_rows.append({
            "Name": r["customer_name"],
            "Email": r["email"],
            "Subject": r["subject"],
            "Body": r["body"],
        })

    st.markdown("")
    st.download_button(
        "📥 Export all emails as CSV",
        data=pd.DataFrame(export_rows).to_csv(index=False),
        file_name="churnguard_emails.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True
    )

    st.markdown("""
<div class="footer">
ChurnGuard · Built in 3 hours, AI-assisted · Groq + Llama 3.3 70B ·
Deploy free at <a href="https://share.streamlit.io" style="color:#4a9eff;">share.streamlit.io</a>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
