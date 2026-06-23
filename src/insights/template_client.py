"""
Template-based insight generator.

Produces natural-language analysis from computed model results without
requiring any external API. All logic is deterministic conditional
branching on real values — no random text generation.
"""
from __future__ import annotations

from src.insights.base import BaseInsightClient
from src.insights.models import InsightData, InsightReport

# ── Feature name → plain English ─────────────────────────────────────────────
_FEATURE_NAMES: dict[str, str] = {
    "recency_days": "time since last purchase",
    "frequency": "purchase frequency",
    "monetary_total": "total lifetime spend",
    "aov": "average order value",
    "tenure_days": "length of customer relationship",
    "tx_per_month": "monthly transaction rate",
    "gap_std_days": "consistency of purchase timing",
}

_FEATURE_EXPLANATIONS: dict[str, str] = {
    "recency_days": (
        "Customers who haven't purchased recently are far more likely to lapse — "
        "extended silence is typically the first sign of disengagement."
    ),
    "frequency": (
        "Low-frequency buyers show the highest churn risk. "
        "Customers who purchase consistently tend to maintain their subscriptions."
    ),
    "monetary_total": (
        "Customers with lower lifetime spend are more likely to churn. "
        "Higher investment in the platform tends to correlate with stronger retention."
    ),
    "aov": (
        "Smaller average order sizes correlate with higher churn. "
        "Customers who spend more per transaction are typically more committed."
    ),
    "tenure_days": (
        "Newer customers carry the highest churn risk. "
        "The longer a customer has been active, the more likely they are to stay."
    ),
    "tx_per_month": (
        "Customers who transact infrequently are more vulnerable. "
        "A consistent purchase cadence is one of the strongest signals of loyalty."
    ),
    "gap_std_days": (
        "Irregular purchase timing is a churn warning sign. "
        "Customers with unpredictable buying patterns are harder to retain."
    ),
}


def _health_summary(data: InsightData) -> str:
    snap = data.kpi_snapshot
    cl = data.churn_label_result
    ts = data.kpi_ts
    window = data.churn_window_days
    cr = cl.churn_rate

    if cr == 0.0:
        churn_sentence = (
            f"No customers have lapsed in the last {window} days — "
            "either retention is exceptional or the transaction history is still short."
        )
    elif cr < 0.05:
        n_in = max(2, round(1 / cr))
        churn_sentence = (
            f"Retention is strong: only {cr:.1%} of customers have gone inactive "
            f"in the last {window} days — fewer than 1 in {n_in}."
        )
    elif cr < 0.12:
        churn_sentence = (
            f"Churn sits at {cr:.1%}, which is within an acceptable range "
            f"for subscription businesses over a {window}-day window."
        )
    elif cr < 0.20:
        churn_sentence = (
            f"Churn is at {cr:.1%} — not alarming, but there is clear room "
            f"to improve retention over the {window}-day window."
        )
    elif cr < 0.30:
        n_in = max(2, round(1 / cr))
        churn_sentence = (
            f"Churn is elevated at {cr:.1%}, meaning roughly 1 in {n_in} customers "
            f"has gone inactive within the {window}-day window."
        )
    else:
        churn_sentence = (
            f"Churn is high at {cr:.1%} — more than {cr:.0%} of customers "
            f"have not returned within {window} days. This warrants urgent attention."
        )

    base_sentence = (
        f"The base currently shows {snap.active_subscribers:,} active subscribers "
        f"generating ${snap.mrr:,.0f} in monthly recurring revenue "
        f"(${snap.arpu:.2f} average per customer)."
    )

    rev = ts.monthly_revenue
    if len(rev) >= 6:
        recent_avg = float(rev.iloc[-3:].mean())
        prior_avg = float(rev.iloc[-6:-3].mean())
        pct = (recent_avg - prior_avg) / prior_avg if prior_avg > 0 else 0.0
        if pct > 0.05:
            trend_sentence = f"Revenue has trended upward over recent months (+{pct:.0%}), a positive signal."
        elif pct < -0.05:
            trend_sentence = (
                f"Revenue has dipped {abs(pct):.0%} compared to the prior period — worth monitoring closely."
            )
        else:
            trend_sentence = "Revenue has been broadly stable across recent months."
    elif len(rev) >= 2:
        first_v = float(rev.iloc[0])
        last_v = float(rev.iloc[-1])
        if last_v > first_v * 1.05:
            trend_sentence = "Revenue shows an upward trend over the available history."
        elif last_v < first_v * 0.95:
            trend_sentence = "Revenue shows a downward trend over the available history."
        else:
            trend_sentence = "Revenue has remained relatively stable over the available history."
    else:
        trend_sentence = ""

    parts = [churn_sentence, base_sentence]
    if trend_sentence:
        parts.append(trend_sentence)
    return " ".join(parts)


def _churn_analysis(data: InsightData) -> str:
    cl = data.churn_label_result
    shap = data.shap_result
    rfm = data.rfm_result
    window = data.churn_window_days
    n_total = cl.n_churned + cl.n_active

    if shap is not None and len(shap.feature_importance) >= 1:
        feat_col = "feature_name" if "feature_name" in shap.feature_importance.columns else shap.feature_importance.columns[0]
        top_feat = str(shap.feature_importance[feat_col].iloc[0])
        top_name = _FEATURE_NAMES.get(top_feat, top_feat.replace("_", " "))
        top_exp = _FEATURE_EXPLANATIONS.get(top_feat, "")

        lead = f"The model's strongest churn signal is **{top_name}**. {top_exp}"

        if len(shap.feature_importance) >= 2:
            sec_feat = str(shap.feature_importance[feat_col].iloc[1])
            sec_name = _FEATURE_NAMES.get(sec_feat, sec_feat.replace("_", " "))
            sec_exp = _FEATURE_EXPLANATIONS.get(sec_feat, "")
            secondary = f"The second-strongest predictor is **{sec_name}**. {sec_exp}"
        else:
            secondary = ""

        context = (
            f"Of {n_total:,} customers analysed, {cl.n_churned:,} ({cl.churn_rate:.1%}) "
            f"are classified as churned based on {window} days of inactivity."
        )

        parts = [lead]
        if secondary:
            parts.append(secondary)
        parts.append(context)
        return " ".join(parts)

    else:
        base = (
            f"Of {n_total:,} customers analysed, {cl.n_churned:,} ({cl.churn_rate:.1%}) "
            f"have not made a purchase in the last {window} days."
        )
        if rfm is not None:
            feat = rfm.features
            avg_recency = float(feat["recency_days"].mean())
            avg_freq = float(feat["frequency"].mean())
            rfm_context = (
                f"Across the active customer base, average recency is {avg_recency:.0f} days "
                f"and customers transact an average of {avg_freq:.1f} times. "
                "Train the churn model on the Predictions page to identify which specific "
                "behaviours are driving lapse."
            )
            return f"{base} {rfm_context}"
        return base


def _revenue_outlook(data: InsightData) -> str:
    snap = data.kpi_snapshot
    fb = data.forecast_bundle
    ts = data.kpi_ts

    if fb is not None:
        rev = fb.revenue
        subs = fb.subscribers
        end_rev = float(rev.forecast.iloc[-1])
        total_fc = float(rev.forecast.sum())
        start_rev = float(rev.historical.iloc[-1]) if len(rev.historical) > 0 else snap.mrr

        if end_rev > start_rev * 1.02:
            direction = "grow"
        elif end_rev < start_rev * 0.98:
            direction = "decline"
        else:
            direction = "remain stable"

        forecast_sentence = (
            f"The {rev.horizon_months}-month revenue forecast projects monthly revenue to **{direction}**, "
            f"reaching ${end_rev:,.0f}/month by end of horizon "
            f"(${total_fc:,.0f} total over the period)."
        )

        end_subs = float(subs.forecast.iloc[-1])
        if end_subs > snap.active_subscribers * 1.02:
            sub_sentence = (
                f"Subscriber count is projected to grow to {end_subs:,.0f} "
                f"by month {subs.horizon_months}, up from {snap.active_subscribers:,} today."
            )
        elif end_subs < snap.active_subscribers * 0.98:
            sub_sentence = (
                f"Subscriber count is projected to decline to {end_subs:,.0f} "
                f"by month {subs.horizon_months}, down from {snap.active_subscribers:,} today."
            )
        else:
            sub_sentence = (
                f"Subscriber count is projected to hold near current levels "
                f"({end_subs:,.0f} at month {subs.horizon_months})."
            )

        return f"{forecast_sentence} {sub_sentence}"

    else:
        arpu_sentence = f"Current ARPU is ${snap.arpu:.2f} per subscriber per month."
        rev = ts.monthly_revenue

        if len(rev) >= 4:
            recent_avg = float(rev.iloc[-3:].mean())
            prior_count = max(1, len(rev) - 3)
            prior_avg = float(rev.iloc[:prior_count].mean())
            pct = (recent_avg - prior_avg) / prior_avg if prior_avg > 0 else 0.0
            if pct > 0.05:
                trend = f"Revenue has grown {pct:.0%} over recent months. {arpu_sentence}"
            elif pct < -0.05:
                trend = f"Revenue has declined {abs(pct):.0%} over recent months. {arpu_sentence}"
            else:
                trend = f"Revenue has been broadly stable. {arpu_sentence}"
        else:
            trend = f"Insufficient history for a trend analysis. {arpu_sentence}"

        return (
            f"{trend} Navigate to the Forecasting page to generate a full 12-month projection."
        )


def _customer_segments(data: InsightData) -> str:
    segs = data.segments
    cl = data.churn_label_result

    if segs is not None and len(segs) > 0:
        counts = segs.value_counts()
        total = len(segs)

        lines: list[str] = []

        n_churned = int(counts.get("Churned", 0))
        if n_churned > 0:
            lines.append(
                f"**{n_churned:,} Churned** ({n_churned / total:.0%}) — customers who have lapsed "
                "and are candidates for a win-back campaign."
            )

        n_at_risk = int(counts.get("At Risk", 0))
        if n_at_risk > 0:
            lines.append(
                f"**{n_at_risk:,} At Risk** ({n_at_risk / total:.0%}) — customers the model flags as likely "
                "to churn soon. Targeted outreach now is more cost-effective than re-acquisition later."
            )

        n_new = int(counts.get("New", 0))
        if n_new > 0:
            lines.append(
                f"**{n_new:,} New** ({n_new / total:.0%}) — recent customers whose long-term retention "
                "is still being established. First-90-day engagement programmes have high ROI here."
            )

        n_hv = int(counts.get("High Value", 0))
        if n_hv > 0:
            lines.append(
                f"**{n_hv:,} High Value** ({n_hv / total:.0%}) — top-spending customers "
                "who warrant priority retention attention."
            )

        n_healthy = int(counts.get("Healthy", 0))
        if n_healthy > 0:
            lines.append(
                f"**{n_healthy:,} Healthy** ({n_healthy / total:.0%}) — active, engaged customers "
                "with stable purchasing patterns."
            )

        if not lines:
            return "Segment data is available but no customers were categorised."

        intro = f"Customer base of {total:,} breaks down as follows:\n\n"
        return intro + "\n\n".join(f"- {line}" for line in lines)

    else:
        n_active = cl.n_active
        n_churned_raw = cl.n_churned
        total = n_active + n_churned_raw
        return (
            f"Of {total:,} customers, {n_active:,} ({1 - cl.churn_rate:.1%}) are active and "
            f"{n_churned_raw:,} ({cl.churn_rate:.1%}) have lapsed. "
            "Train the churn model on the Predictions page to get a full segment breakdown "
            "including At Risk, High Value, and New customer groups."
        )


def _recommendations(data: InsightData) -> str:
    cr = data.churn_label_result.churn_rate
    segs = data.segments
    fb = data.forecast_bundle
    snap = data.kpi_snapshot
    clv = data.clv_result

    recs: list[str] = []

    # Rec 1: immediate churn action
    if segs is not None:
        seg_counts = segs.value_counts()
        n_at_risk = int(seg_counts.get("At Risk", 0))
        n_churned_seg = int(seg_counts.get("Churned", 0))
        if n_at_risk > 0:
            recs.append(
                f"**Priority outreach:** Contact the {n_at_risk:,} at-risk customers before they lapse. "
                "Re-engaging an existing customer typically costs 5–25× less than acquiring a new one."
            )
        elif n_churned_seg > 0:
            recs.append(
                f"**Win-back campaign:** Reach out to {n_churned_seg:,} churned customers with a targeted offer. "
                "Even a 10% recovery rate adds meaningfully to revenue."
            )
        else:
            recs.append(
                "**Monitor early warnings:** No customers are currently flagged as at-risk. "
                "Continue running the model monthly to catch emerging churn signals early."
            )
    else:
        if cr > 0.15:
            recs.append(
                f"**Address churn urgently:** At {cr:.1%}, churn is high enough to erode the subscriber base. "
                "Train the churn model to identify which specific customers are at highest risk."
            )
        elif cr > 0.05:
            recs.append(
                f"**Run predictive scoring:** Train the churn model to identify your highest-risk customers "
                f"from the {data.churn_label_result.n_active:,} currently active. "
                "Acting before customers lapse costs less than winning them back."
            )
        else:
            recs.append(
                "**Maintain momentum:** Retention is strong. Continue monitoring monthly and "
                "run the churn model to catch early signals before they become a trend."
            )

    # Rec 2: protect high-value customers
    if clv is not None:
        top_clv = float(clv.clv_per_customer["expected_clv"].quantile(0.9))
        recs.append(
            f"**Protect your top tier:** Your top-decile customers have an expected CLV of ${top_clv:,.0f}+. "
            "Consider a dedicated VIP retention programme — losing even a handful has an outsized "
            "impact on total revenue."
        )
    else:
        recs.append(
            f"**Unlock CLV estimates:** With an ARPU of ${snap.arpu:.2f}, training the model will produce "
            "Kaplan-Meier lifetime value estimates per customer — letting you allocate retention spend "
            "where it has the highest return."
        )

    # Rec 3: forecast or growth action
    if fb is not None:
        end_rev = float(fb.revenue.forecast.iloc[-1])
        start_rev = float(fb.revenue.historical.iloc[-1]) if len(fb.revenue.historical) > 0 else snap.mrr
        if end_rev < start_rev * 0.95:
            recs.append(
                f"**Forecast alert:** Revenue is projected to decline to ${end_rev:,.0f}/month by the "
                "end of the horizon. Review the Forecasting page and consider growth initiatives "
                "to offset projected losses."
            )
        else:
            recs.append(
                "**Sustain the trajectory:** The forecast shows a positive revenue outlook. "
                "Focus retention efforts on at-risk segments to protect this trend — "
                "churn is the fastest way to undermine a positive forecast."
            )
    else:
        recs.append(
            "**Generate a forecast:** Run the Forecasting page for a 12-month revenue and subscriber "
            "projection. This gives you a concrete target to plan against and helps you spot "
            "trajectory risks before they materialise."
        )

    numbered = [f"**{i + 1}.** {rec}" for i, rec in enumerate(recs[:3])]
    return "\n\n".join(numbered)


def _model_confidence(data: InsightData) -> str | None:
    if data.model_metrics is None:
        return None

    m = data.model_metrics
    auc = m.auc
    precision = m.precision

    if auc >= 0.90:
        confidence, detail = (
            "excellent",
            "The model is highly reliable for identifying at-risk customers.",
        )
    elif auc >= 0.80:
        confidence, detail = (
            "strong",
            "The model reliably distinguishes churned from active customers.",
        )
    elif auc >= 0.70:
        confidence, detail = (
            "good",
            "The model is useful for prioritising outreach, though some false positives are expected.",
        )
    elif auc >= 0.60:
        confidence, detail = (
            "moderate",
            "The model provides directional guidance but should not be acted on mechanically.",
        )
    else:
        confidence, detail = (
            "limited",
            "The model is struggling to separate churned from active customers. "
            "More transaction history will improve accuracy.",
        )

    auc_sentence = (
        f"The churn model achieves an AUC of {auc:.3f}, indicating **{confidence}** predictive accuracy. "
        f"{detail}"
    )
    precision_sentence = (
        f"When the model flags a customer as at-risk, it is correct approximately {precision:.0%} of the time "
        f"(precision: {precision:.3f}). Trained on {m.n_train:,} customers, tested on {m.n_test:,}."
    )
    return f"{auc_sentence} {precision_sentence}"


class TemplateInsightClient(BaseInsightClient):
    """
    Generates structured natural-language insights from model results.

    No API key or network access required. All output is derived deterministically
    from the numbers already computed by the analytics and ML layers.
    """

    def generate(self, data: InsightData) -> InsightReport:
        return InsightReport(
            health_summary=_health_summary(data),
            churn_analysis=_churn_analysis(data),
            revenue_outlook=_revenue_outlook(data),
            customer_segments=_customer_segments(data),
            recommendations=_recommendations(data),
            model_confidence=_model_confidence(data),
            client_type="template",
        )
