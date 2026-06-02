# ChurnGuard — AI Retention Agent

An AI-powered retention agent for subscription businesses. Upload your customer list, get personalized re-engagement emails for your highest-risk, highest-value customers — in under 60 seconds.

**Live demo:** [subscription-retention-agent-moiz.streamlit.app](https://subscription-retention-agent-moiz.streamlit.app)

**No install required** — runs in your browser via Streamlit Cloud.

---

## What it does

1. Upload a CSV of your customers — or click **▶ Try sample data** instantly, no API key needed
2. Scores every customer using a **weighted RFM model** (Recency 50%, Frequency 30%, Monetary 20%) — churn risk shown as a progress bar 0–100%
3. Filters to only **high-risk + high-LTV** customers — the ones worth spending retention budget on (emailing everyone wastes budget and trains low-value customers to wait for discounts)
4. One click generates a **personalised re-engagement email** per customer using Groq AI, with the agent's reasoning shown so you can review before sending
5. Export all emails as CSV — paste into Gmail, Outlook, Mailchimp, whatever your team already uses

---

## Deploy your own (2 minutes, free)

### Option A — Streamlit Cloud (recommended, no install)

1. Fork this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your forked repo → set main file to `app.py`
4. Click **Deploy** — you get a public URL in ~60 seconds
5. *(Optional but recommended)* Go to your app → **Settings** → **Secrets** and add:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```
This pre-fills the API key for anyone who opens the link — zero friction for your team.

6. Share the URL with your team. They just open it in a browser — no install, no signup.

### Option B — Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## CSV format

Your CSV needs these columns:

| Column | Example |
|--------|---------|
| `customer_name` | Sarah Mitchell |
| `email` | sarah@example.com |
| `last_purchase_date` | 2026-03-10 |
| `total_orders` | 12 |
| `total_spend` | 1840.00 |
| `product_category` | Skincare |

Download a sample CSV from the app sidebar.

---

## Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free) → API Keys → Create key
3. Paste it into the app sidebar

**Why Groq and not ChatGPT or Claude?**
- **Free tier, no credit card** — Groq's free tier is enough to run this tool for hundreds of customers
- **Same output quality** — runs Llama 3.3 70B, comparable to paid frontier models for email generation
- **Much faster** — Groq's hardware inference is 10x faster than OpenAI/Anthropic APIs for bulk generation
- Claude and ChatGPT require a paid API subscription to use programmatically — a barrier for most teams

---

## How the scoring works

ChurnGuard uses a **weighted RFM model**:

- **Recency (50%)** — days since last purchase (most predictive of churn)
- **Frequency (30%)** — total number of orders
- **Monetary (20%)** — total spend

Churn risk score = inverse of RFM score, scaled 0–100%.

**Only high-LTV customers (total spend ≥ $500) are targeted for email outreach.** This is intentional — sending retention emails to low-value customers wastes budget and trains them to wait for discounts.

---

## Built with

- [Streamlit](https://streamlit.io) — web UI, deployable to Streamlit Cloud in 2 minutes
- [Groq](https://groq.com) — fast LLM inference (free tier, no credit card)
- [Llama 3.3 70B](https://groq.com) — email generation model
- Python + Pandas — RFM scoring engine

Built in 3 hours, AI-assisted.

---

## One thing AI got wrong during the build

The AI generated the first version of the email prompt without passing enough customer context — it used the customer's name in the opener but then defaulted to "valued customer" for the rest of the body. Every email came out generic and interchangeable.

The fix was straightforward: restructure the prompt to explicitly pass `days_since_purchase`, `product_category`, `total_orders`, and `total_spend` as named fields, and add a rule that the email must reference the product category and the customer's history. After that change the emails became noticeably more specific and usable. The lesson: AI doesn't infer what personalisation means — you have to spell out exactly which data points it should use and how.

---

## One thing cut from the plan and why

The original plan included a chart showing churn trend over time — a line graph of risk scores across cohorts. It was cut because it looked impressive but didn't actually help the person using the tool take any action.

The whole point of ChurnGuard is answering one question: *who do I email today, and what do I say?* A trend chart answers a different question — one that belongs in a quarterly review, not a daily retention workflow. Cutting it kept the tool focused on action rather than analysis.
