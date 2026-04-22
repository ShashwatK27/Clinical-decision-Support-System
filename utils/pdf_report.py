"""
pdf_report.py — Generate a clinical PDF report from a CDSS analysis result.

Usage:
    from utils.pdf_report import generate_pdf_report
    pdf_bytes = generate_pdf_report(report_data)

report_data structure:
    {
        "prescription_text": str,
        "drugs": list[str],
        "dosage_warnings": list[DosageWarning],
        "interactions": list[Interaction],
        "predictions": list[dict],    # {"condition_label", "confidence_pct"}
        "recommendations": dict,      # {condition: rec_text}
        "similar_cases": list[tuple], # [(meta_dict, score), ...]
        "timestamp": str,
    }
"""

from __future__ import annotations
from datetime import datetime
from fpdf import FPDF

# ── Page geometry ──────────────────────────────────────────────────────────────
_LM = 12          # left margin (mm)
_RM = 12          # right margin (mm)
_TM = 24          # top margin (mm, leaves room for header)
_PW = 210         # page width A4
_W  = _PW - _LM - _RM   # usable content width = 186 mm

# ── Colours (R, G, B) ─────────────────────────────────────────────────────────
_NAVY  = (22,  60, 100)
_TEAL  = (0,  128, 128)
_RED   = (192,  0,   0)
_AMBER = (204, 119,  0)
_GREEN = (30,  130,  70)
_GREY  = (110, 110, 110)
_LGREY = (240, 240, 240)
_WHITE = (255, 255, 255)
_BLACK = (30,  30,  30)


def _safe(text: str) -> str:
    """Replace characters outside latin-1 so fpdf2 Helvetica never crashes."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


class _ClinicalPDF(FPDF):
    """FPDF subclass — branded header/footer, helper drawing methods."""

    # ── System hooks ──────────────────────────────────────────────────────────
    def header(self):
        self.set_fill_color(*_NAVY)
        self.rect(0, 0, 210, 18, "F")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*_WHITE)
        self.set_xy(_LM, 4)
        self.cell(_W, 7, "Clinical Decision Support System - Analysis Report", ln=False)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(180, 200, 220)
        self.set_xy(_LM, 12)
        self.cell(_W, 4, "CDSS v2.0  |  For clinical decision support only - not a substitute for professional judgement.")
        self.ln(10)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*_GREY)
        self.cell(_W, 8, f"Page {self.page_no()} | CDSS Confidential", align="C")

    # ── Drawing helpers ───────────────────────────────────────────────────────
    def _reset(self):
        """Reset cursor to left margin, black text."""
        self.set_x(_LM)
        self.set_text_color(*_BLACK)

    def section_title(self, title: str):
        self._reset()
        self.set_fill_color(*_TEAL)
        self.set_text_color(*_WHITE)
        self.set_font("Helvetica", "B", 11)
        self.cell(_W, 8, f"  {_safe(title)}", ln=True, fill=True)
        self._reset()
        self.ln(2)

    def body(self, text: str, bold: bool = False):
        self._reset()
        self.set_font("Helvetica", "B" if bold else "", 10)
        self.multi_cell(_W, 6, _safe(text))

    def badge_row(self, severity: str, label: str):
        """Coloured severity pill + label text on one line."""
        colours = {
            "severe":   _RED,   "high":     _RED,
            "moderate": _AMBER, "caution":  _AMBER,
            "mild":     _GREEN, "ok":       _GREEN,
        }
        pill_w = 30
        lbl_w  = _W - pill_w
        colour = colours.get(severity.lower(), _GREY)
        self._reset()
        self.set_fill_color(*colour)
        self.set_text_color(*_WHITE)
        self.set_font("Helvetica", "B", 9)
        self.cell(pill_w, 7, _safe(severity.upper()), fill=True, align="C")
        self.set_text_color(*_BLACK)
        self.set_font("Helvetica", "", 10)
        self.cell(lbl_w, 7, _safe(f"  {label}"), ln=True)
        self._reset()

    def note_line(self, text: str):
        """Small grey italic note, always from the left margin."""
        self._reset()
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(*_GREY)
        self.multi_cell(_W, 5, _safe(f"    {text}"))
        self._reset()

    def kv(self, key: str, value: str):
        self._reset()
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*_NAVY)
        self.cell(50, 6, _safe(key + ":"))
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*_BLACK)
        self.multi_cell(_W - 50, 6, _safe(value))
        self._reset()

    def divider(self):
        self.set_draw_color(*_TEAL)
        self.set_line_width(0.3)
        self.line(_LM, self.get_y(), _PW - _RM, self.get_y())
        self.ln(3)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_pdf_report(data: dict) -> bytes:
    """
    Generate a 2-page clinical PDF report and return as bytes.
    Safe to call from st.download_button() directly.
    """
    pdf = _ClinicalPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(_LM, _TM, _RM)
    pdf.add_page()

    timestamp = data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # ── Meta bar ──────────────────────────────────────────────────────────────
    pdf._reset()
    pdf.set_fill_color(*_LGREY)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*_GREY)
    pdf.cell(_W, 6,
             _safe(f"Generated: {timestamp}    |    Review all output with a qualified clinician."),
             ln=True, fill=True)
    pdf.ln(3)

    # ── Prescription input ────────────────────────────────────────────────────
    pdf.section_title("Prescription Input")
    pdf._reset()
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_fill_color(250, 250, 250)
    pdf.set_text_color(50, 50, 80)
    pdf.multi_cell(_W, 6, _safe(data.get("prescription_text", "-")), fill=True, border=1)
    pdf._reset()
    pdf.ln(3)

    # ── Detected drugs ────────────────────────────────────────────────────────
    pdf.section_title("Detected Drugs")
    drugs = data.get("drugs", [])
    pdf.body(", ".join(drugs) if drugs else "No drugs detected.")
    pdf.ln(3)

    # ── Dosage validation ─────────────────────────────────────────────────────
    dose_warnings = data.get("dosage_warnings", [])
    pdf.section_title("Dosage Validation")
    if not dose_warnings:
        pdf.body("All detected doses are within safe reference ranges.")
    else:
        for dw in dose_warnings:
            pdf.badge_row(
                dw.severity,
                f"{dw.drug.upper()}: {dw.dose_value} {dw.dose_unit} "
                f"(max {dw.limit_type}: {dw.limit_value} {dw.dose_unit})",
            )
            pdf.note_line(dw.note)
            pdf.ln(1)
    pdf.ln(2)

    # ── Drug interactions ─────────────────────────────────────────────────────
    interactions = data.get("interactions", [])
    pdf.section_title("Drug Interaction Check")
    if not interactions:
        pdf.body("No known drug-drug interactions detected.")
    else:
        for ix in interactions:
            pair = " + ".join(sorted(ix.drugs))
            pdf.badge_row(ix.severity, pair)
            pdf.note_line(f"Effect: {ix.effect}")
            pdf.note_line(f"Action: {ix.recommendation}")
            pdf.ln(2)
    pdf.ln(2)

    # ── Condition predictions ─────────────────────────────────────────────────
    predictions = data.get("predictions", [])
    pdf.section_title("Predicted Clinical Conditions")
    if not predictions:
        pdf.body("No conditions predicted with sufficient confidence.")
    else:
        # Table header
        pdf._reset()
        pdf.set_fill_color(*_NAVY)
        pdf.set_text_color(*_WHITE)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(140, 7, "Condition", fill=True, border=1)
        pdf.cell(_W - 140, 7, "Confidence", fill=True, border=1, ln=True)

        for i, pred in enumerate(predictions):
            pdf._reset()
            fill_clr = (240, 248, 255) if i % 2 == 0 else (255, 255, 255)
            pdf.set_fill_color(*fill_clr)
            pdf.set_text_color(*_BLACK)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(140, 7, _safe(pred.get("condition_label", "")), fill=True, border=1)
            pdf.cell(_W - 140, 7, f"{pred.get('confidence_pct', '?')}%",
                     fill=True, border=1, align="C", ln=True)
    pdf.ln(3)

    # ── Clinical recommendations ──────────────────────────────────────────────
    recs = data.get("recommendations", {})
    pdf.section_title("Clinical Recommendations")
    if not recs:
        pdf.body("No recommendations available.")
    else:
        for i, (cond, rec) in enumerate(recs.items()):
            pdf._reset()
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*_NAVY)
            pdf.cell(_W, 6, _safe(cond.title()), ln=True)
            pdf._reset()
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(_W, 6, _safe(f"    {rec}"))
            pdf.ln(1)
    pdf.ln(3)

    # ── Similar historical cases ──────────────────────────────────────────────
    similar = data.get("similar_cases", [])
    pdf.section_title("Similar Historical Cases")
    if not similar:
        pdf.body("No similar cases found above the similarity threshold.")
    else:
        for i, (meta, score) in enumerate(similar, 1):
            drugs_str = ", ".join(meta.get("drugs", []))
            conds_str = ", ".join(meta.get("conditions", [])) or "-"
            pdf._reset()
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(_W, 6, _safe(f"Case {i}  -  Similarity: {score:.1%}"), ln=True)
            pdf.note_line(f"Drugs: {drugs_str}")
            pdf.note_line(f"Conditions: {conds_str}")
            pdf.divider()

    # ── Disclaimer page ───────────────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("Important Disclaimer")
    pdf._reset()
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(_W, 6, _safe(
        "This report was generated automatically by the Clinical Decision Support "
        "System (CDSS) and is intended solely as a decision-support aid for "
        "qualified healthcare professionals. It must NOT be used as a standalone "
        "basis for clinical decisions.\n\n"
        "- Drug interaction data is sourced from a curated local database and may "
        "not be exhaustive.\n"
        "- Dosage thresholds are reference values and may vary by patient weight, "
        "renal function, age, comorbidities, and local prescribing guidelines.\n"
        "- Condition predictions are based on vector similarity and rule-based "
        "reasoning; they are probabilistic, not diagnostic.\n\n"
        "Always consult current prescribing guidelines (BNF, MIMS, or equivalent) "
        "and exercise independent clinical judgement."
    ))

    return bytes(pdf.output())
