import streamlit as st
import pandas as pd
import json
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
                 padding: 20px 24px; text-align: center; }
    .stat-num  { font-size: 2rem; font-weight: 800; color: #4a9eff; }
    .stat-lbl  { font-size: 0.8rem; color: #6b7280; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
    .risk-high   { color: #ff4b4b; font-weight: 700; }
    .risk-medium { color: #ffa500; font-weight: 700; }
    .risk-low    { color: #00c853; font-weight: 700; }
    .email-card  { background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 12px;
                   padding: 24px; margin-bottom: 16px; }
    .reasoning   { background: #0d1117; border-left: 3px solid #4a9eff; border-radius: 0 8px 8px 0;
                   padding: 12px 16px; color: #7a8599; font-size: 0.85rem; margin-bottom: 16px; line-height: 1.6; }
    .email-body  { background: #12151f; border: 1px solid #2a2d3a; border-radius: 8px;
                   padding: 16px; font-family: 'Georgia', serif; font-size: 0.9rem;
                   color: #d1d5db; line-height: 1.8; white-space: pre-wrap; }
    .subject-line { font-size: 0.95rem; color: #4a9eff; font-weight: 600; margin-bottom: 12px; }
    .badge-high   { background:#ff4b4b22; border:1px solid #ff4b4b; border-radius:20px;
                    padding:2px 10px; color:#ff4b4b; font-size:0.75rem; font-weight:700; }
    .badge-medium { background:#ffa50022; border:1px solid #ffa500; border-radius:20px;
                    padding:2px 10px; color:#ffa500; font-size:0.75rem; font-weight:700; }
    .badge-low    { background:#00c85322; border:1px solid #00c853; border-radius:20px;
                    padding:2px 10px; color:#00c853; font-size:0.75rem; font-weight:700; }
    .step-box { background:#1a1d27; border:1px solid #2a2d3a; border-radius:12px; padding:20px 24px; margin-bottom:12px; }
    .step-num { color:#4a9eff; font-weight:800; font-size:1.1rem; }
    .why-box  { background:#0d1117; border:1px solid #2a2d3a; border-radius:10px; padding:16px 20px; margin-top:12px; }
    .footer   { text-align:center; color:#4a5568; font-size:0.78rem; margin-top:48px; padding-top:24px;
                border-top:1px solid #2a2d3a; }
    h2, h3 { color: #f0f2f5 !important; }
    .stButton>button { border-radius:8px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


SAMPLE_CSV = """customer_name,email,last_purchase_date,total_orders,total_spend,product_category
Sarah Mitchell,sarah@example.com,2024-11-03,12,1840.00,Skincare
James Okafor,james@example.com,2025-04-20,3,210.00,Supplements
Priya Nair,priya@example.com,2025-05-15,8,960.00,Fitness
Tom Bergmann,tom@example.com,2024-09-12,2,95.00,Supplements
Elena Vasquez,elena@example.com,2025-05-28,15,2200.00,Skincare
David Chen,david@example.com,2024-12-01,5,430.00,Fitness
Aisha Patel,aisha@example.com,2025-03-10,1,45.00,Supplements
Marcus Webb,marcus@example.com,2025-01-22,7,880.00,Skincare
Fatima Al-Hassan,fatima@example.com,2024-10-05,4,320.00,Fitness
Ryan Kowalski,ryan@example.com,2025-05-01,9,1100.00,Supplements
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
    system_prompt = """You are a retention specialist for a subscription business.
Write ONE short, warm, personal re-engagement email for an at-risk customer.

Rules:
- Use their first name only
- Reference their specific product category naturally
- Lead with VALUE — a tip, insight, or something they are missing — not "we miss you"
- Never use the words: churn, retention, "we noticed you haven't", "it's been a while"
- Keep it under 120 words
- Subject line must be specific to their category, not generic
- Sign off as "The [Category] Team"

Respond ONLY in this exact JSON format:
{
  "reasoning": "2-3 sentences: why this customer is at risk and what angle makes sense for them specifically",
  "subject": "email subject line",
  "body": "full email body"
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
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"reasoning": "Could not parse response.", "subject": "Checking in", "body": raw}


def main():

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Setup")

        st.markdown("**Step 1 — Get a free Groq API key**")
        st.markdown("""
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up — free, no credit card
3. Click **API Keys** → **Create key**
4. Copy and paste it below
""")
        st.markdown("""
<div class="why-box">
<strong>Why Groq and not ChatGPT or Claude?</strong><br><br>
ChatGPT and Claude both require a <em>paid</em> API subscription to use in tools like this.
Groq is free, runs Llama 3.3 70B (same quality), and is significantly faster for bulk generation.
</div>
""", unsafe_allow_html=True)

        st.markdown("")
        groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

        st.markdown("---")
        st.markdown("**Step 2 — Choose model**")
        model = st.selectbox("Model", [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ], index=0)

        st.markdown("---")
        st.markdown("**CSV columns required**")
        st.markdown("`customer_name` `email` `last_purchase_date` `total_orders` `total_spend` `product_category`")
        st.markdown("")
        st.download_button("📥 Download sample CSV", data=SAMPLE_CSV,
                           file_name="sample_customers.csv", mime="text/csv",
                           use_container_width=True)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="hero">
  <h1>🛡️ ChurnGuard</h1>
  <p>Upload your customer list. ChurnGuard scores every customer for churn risk, identifies the ones worth spending retention budget on, and writes a personalised re-engagement email for each — in under 60 seconds.<br><br>
  Built for subscription businesses. No engineering required.</p>
</div>
""", unsafe_allow_html=True)

    # ── Upload ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded = st.file_uploader("Upload your customer CSV", type=["csv"], label_visibility="collapsed")
    with col2:
        use_sample = st.button("▶ Try with sample data", use_container_width=True, type="secondary")

    df_raw = None
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_raw)} customers.")
    elif use_sample:
        df_raw = pd.read_csv(io.StringIO(SAMPLE_CSV))
        st.info("Using sample data — 10 customers across Skincare, Fitness, and Supplements.")

    # ── Landing instructions (shown before any data) ──────────────────────────
    if df_raw is None:
        st.markdown("---")
        st.markdown("### How it works")
        steps = [
            ("1", "Upload your CSV", "Use the uploader above, or click <strong>Try with sample data</strong> to see it running immediately. No API key needed for scoring."),
            ("2", "Customers are scored", "ChurnGuard runs every customer through a weighted RFM model — Recency (50%), Frequency (30%), Monetary (20%) — and assigns a churn risk score 0–100%."),
            ("3", "High-value customers are flagged", "Only customers with <strong>high lifetime value AND high churn risk</strong> are surfaced for outreach. Emailing everyone wastes budget and trains low-value customers to wait for discounts."),
            ("4", "AI writes the emails", "Paste your free Groq API key in the sidebar. One click generates a personalised email per customer, with the agent's reasoning shown so you can review before sending."),
            ("5", "Export and send", "Download all emails as CSV or copy individually. Paste into Gmail, Outlook, Mailchimp — whatever your team already uses."),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
<div class="step-box">
<span class="step-num">Step {num}</span> &nbsp; <strong>{title}</strong><br>
<span style="color:#9ca3b0; font-size:0.9rem;">{desc}</span>
</div>
""", unsafe_allow_html=True)
        st.markdown("""
<div class="footer">
Built in 3 hours · AI-assisted (Groq + Llama 3.3 70B) · Deployable to Streamlit Cloud in 2 minutes
</div>
""", unsafe_allow_html=True)
        return

    # ── Validate columns ──────────────────────────────────────────────────────
    required = {"customer_name", "email", "last_purchase_date", "total_orders", "total_spend", "product_category"}
    missing = required - set(df_raw.columns)
    if missing:
        st.error(f"Missing columns in your CSV: {', '.join(sorted(missing))}")
        return

    df = score_customers(df_raw)

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("---")
    total = len(df)
    high = len(df[df["risk_level"] == "High"])
    medium = len(df[df["risk_level"] == "Medium"])
    actionable = len(df[(df["risk_level"].isin(["High", "Medium"])) & (df["ltv_tier"] == "High")])

    c1, c2, c3, c4 = st.columns(4)
    for col, num, lbl in [
        (c1, total, "Total Customers"),
        (c2, high, "High Risk"),
        (c3, medium, "Medium Risk"),
        (c4, actionable, "High-Value At Risk"),
    ]:
        col.markdown(f"""
<div class="stat-card">
  <div class="stat-num">{num}</div>
  <div class="stat-lbl">{lbl}</div>
</div>""", unsafe_allow_html=True)

    # ── Risk table ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Customer Risk Scores")

    f1, f2 = st.columns(2)
    with f1:
        risk_filter = st.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium"])
    with f2:
        ltv_filter = st.multiselect("LTV tier", ["High", "Medium", "Low"], default=["High", "Medium"])

    df_filtered = df[df["risk_level"].isin(risk_filter) & df["ltv_tier"].isin(ltv_filter)]

    display_cols = ["customer_name", "email", "product_category", "days_since_purchase",
                    "total_orders", "total_spend", "churn_risk_score", "risk_level", "ltv_tier"]

    st.dataframe(
        df_filtered[display_cols].rename(columns={
            "customer_name": "Name", "email": "Email", "product_category": "Category",
            "days_since_purchase": "Days Since Purchase", "total_orders": "Orders",
            "total_spend": "Spend ($)", "churn_risk_score": "Churn Risk %",
            "risk_level": "Risk", "ltv_tier": "LTV Tier"
        }),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Churn Risk %": st.column_config.ProgressColumn(
                "Churn Risk %", min_value=0, max_value=100, format="%.1f%%"
            ),
            "Spend ($)": st.column_config.NumberColumn("Spend ($)", format="$%.2f"),
        }
    )

    st.download_button("📥 Export scored data (CSV)",
                       data=df_filtered[display_cols].to_csv(index=False),
                       file_name="churnguard_scored.csv", mime="text/csv")

    # ── Email generation ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Generate Retention Emails")

    targets = df[(df["risk_level"].isin(["High", "Medium"])) & (df["ltv_tier"] == "High")]

    if len(targets) == 0:
        st.info("No high-value at-risk customers found with current filters.")
        return

    st.markdown(f"**{len(targets)} customers** flagged for retention outreach.")
    st.caption("Only high-LTV customers are targeted. Sending to everyone trains low-value customers to expect discounts and dilutes your retention budget.")

    if not groq_key:
        st.warning("Paste your Groq API key in the sidebar to generate emails. It's free — instructions are in the sidebar.")
        return

    if st.button(f"🤖 Generate emails for {len(targets)} customers", type="primary", use_container_width=True):
        client = Groq(api_key=groq_key)
        results = []
        bar = st.progress(0, text="Agent starting...")
        for i, (_, row) in enumerate(targets.iterrows()):
            bar.progress(i / len(targets), text=f"Analysing {row['customer_name']}...")
            try:
                out = generate_retention_email(client, row.to_dict(), model)
                results.append({**row.to_dict(), **out})
            except Exception as e:
                results.append({**row.to_dict(), "reasoning": str(e), "subject": "Error", "body": str(e)})
        bar.progress(1.0, text=f"Done — {len(results)} emails generated.")
        st.session_state["email_results"] = results

    # ── Email results ─────────────────────────────────────────────────────────
    if "email_results" not in st.session_state:
        return

    results = st.session_state["email_results"]
    st.markdown(f"### Emails ({len(results)})")

    export_rows = []
    for r in results:
        badge = f'<span class="badge-{r["risk_level"].lower()}">{r["risk_level"]} Risk</span>'
        st.markdown(f"""
<div class="email-card">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
    <div>
      <strong style="font-size:1rem; color:#f0f2f5;">{r['customer_name']}</strong>
      &nbsp;<span style="color:#6b7280; font-size:0.85rem;">{r['email']}</span>
    </div>
    <div>{badge} &nbsp; <span style="color:#6b7280; font-size:0.8rem;">{r['days_since_purchase']} days since purchase · ${float(r['total_spend']):.0f} LTV</span></div>
  </div>
  <div class="reasoning">🧠 <strong>Why this email:</strong> {r['reasoning']}</div>
  <div class="subject-line">Subject: {r['subject']}</div>
  <div class="email-body">{r['body']}</div>
</div>
""", unsafe_allow_html=True)

        export_rows.append({
            "Name": r["customer_name"], "Email": r["email"],
            "Risk Level": r["risk_level"], "Churn Risk %": r["churn_risk_score"],
            "Days Since Purchase": r["days_since_purchase"],
            "Agent Reasoning": r["reasoning"],
            "Email Subject": r["subject"], "Email Body": r["body"],
        })

    st.markdown("")
    st.download_button(
        "📥 Export all emails (CSV)",
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
