import google.generativeai as genai
import pandas as pd
import os
import csv
import re
import io
import warnings
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()
from src.utils.pii_scrubber import PIIScrubber

class StewardAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Steward Agent with Gemini 3 Flash.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API Key is required.")
        
        genai.configure(api_key=self.api_key)
        
        self.model = genai.GenerativeModel(
            model_name="gemini-3-flash-preview",
            generation_config={"temperature": 0.2}
        )
        self.last_prompt = None
        self.last_response = None

    def analyze_data(self, data_path: str, business_objective: str = "") -> Dict[str, Any]:
        """
        Analyzes the CSV file and generates a dense textual summary.
        Context-aware: audits based on the business_objective.
        Robustness V3: Implements automatic dialect detection and smart profiling.
        """
        # 1. Detect Encoding
        encodings = ['utf-8', 'latin-1', 'cp1252']
        detected_encoding = 'utf-8' # Default
        
        for enc in encodings:
            try:
                with open(data_path, 'r', encoding=enc) as f:
                    f.read(4096)
                detected_encoding = enc
                break
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        
        # 2. Detect Dialect (Robust V3)
        dialect_info = self._detect_csv_dialect(data_path, detected_encoding)
        sep = dialect_info['sep']
        decimal = dialect_info['decimal']
        print(f"Steward Detected: Sep='{sep}', Decimal='{decimal}', Encoding='{detected_encoding}'")

        try:
            # 3. Load Data with Fallbacks & Sampling
            file_size = os.path.getsize(data_path)
            file_size_mb = file_size / (1024 * 1024)
            SAMPLE_SIZE = 5000
            
            # Primary Load Attempt
            try:
                if file_size_mb > 10:
                    print(f"Steward: Sampling {SAMPLE_SIZE} rows (File size: {file_size_mb:.2f}MB)")
                    df = pd.read_csv(data_path, sep=sep, decimal=decimal, encoding=detected_encoding, nrows=SAMPLE_SIZE)
                    was_sampled = True
                else:
                    df = pd.read_csv(data_path, sep=sep, decimal=decimal, encoding=detected_encoding)
                    was_sampled = False
            except Exception as e:
                print(f"Steward: Primary load failed ({e}). Attempting fallback engine...")
                # Fallback: Python engine is slower but more robust
                df = pd.read_csv(data_path, sep=sep if sep else None, decimal=decimal, 
                               encoding=detected_encoding, engine='python', on_bad_lines='skip')
                was_sampled = False # Can't guarantee sampling in fallback generic mode easily without nrows, but usually fine
            
            # 4. Preserve raw headers & Scrub
            df.columns = [str(c) for c in df.columns]
            pii_findings = detect_pii_findings(df)
            scrubber = PIIScrubber()
            df = scrubber.scrub_dataframe(df)

            # 5. Smart Profiling (V3)
            profile = self._smart_profile(df, business_objective)
            shape = df.shape
            dataset_profile = build_dataset_profile(
                df=df,
                objective=business_objective,
                dialect_info=dialect_info,
                encoding=detected_encoding,
                file_size_bytes=file_size,
                was_sampled=was_sampled,
                sample_size=SAMPLE_SIZE if was_sampled else shape[0] if len(df) > 0 else 0,
                pii_findings=pii_findings,
            )
            try:
                write_dataset_profile(dataset_profile)
            except Exception:
                pass
            
            # 6. Construct Prompt
            metadata_str = f"""
            Rows: {shape[0]} (Estimated/Sampled: {was_sampled}), Columns: {shape[1]}
            Filesize: {file_size_mb:.2f} MB
            
            KEY COLUMNS (Top 50 Importance):
            {profile['column_details']}
            
            AMBIGUITY REPORT:
            {profile['ambiguities']}
            
            COLUMN GLOSSARY (Heuristic Hints):
            {profile['glossary']}
            
            {profile['alerts']}
            
            Potential IDs: {profile['ids']}
            Potential Dates: {profile['dates']}
            Target Candidates: {profile['targets']}
            
            Example Rows (Random Sample):
            {profile['examples']}
            """
            
            from src.utils.prompting import render_prompt
            
            SYSTEM_PROMPT_TEMPLATE = """
            You are the Senior Data Steward.
            
            MISSION: Support the Business Objective: "$business_objective"
            
            INPUT DATA PROFILE:
            $metadata_str
            
            INSTRUCTIONS:
            1. Start strictly with "DATA SUMMARY:".
            2. Infer the Business Domain (e.g., Retail, CRM, Manufacturing) based on column names.
            3. Explain the *meaning* of key variables relative to the Objective: "$business_objective".
            4. Highlight Data Quality Blockers and Ambiguities (e.g., numeric-looking strings with commas, percent signs, mixed types).
            5. Mention which columns seem to be Identifiers vs Dates vs Numerical Features, and why they matter to the objective.
            6. Explicitly call out any columns whose meaning is unclear or overloaded.
            7. IF "Sampled: True" is in the profile, YOU MUST EXPLICITLY STATE: "Note: Analysis based on a sample of the first 5000 rows."
            8. Be concise. NO markdown tables. Plain text only.
            """
            
            system_prompt = render_prompt(
                SYSTEM_PROMPT_TEMPLATE,
                business_objective=business_objective,
                metadata_str=metadata_str
            )
            self.last_prompt = system_prompt

            response = self.model.generate_content(system_prompt)
            summary = (getattr(response, "text", "") or "").strip()
            self.last_response = summary

            # Diagnostic logging for empty responses (best-effort, no PII)
            try:
                text_len = len(getattr(response, "text", "") or "")
                candidates = getattr(response, "candidates", None)
                cand_count = len(candidates) if candidates is not None else 0
                first = candidates[0] if cand_count else None
                finish_reason = getattr(first, "finish_reason", None) if first else None
                safety_ratings = getattr(first, "safety_ratings", None) if first else None
                citation_metadata = getattr(first, "citation_metadata", None) if first else None
                citations = getattr(first, "citations", None) if first else None
                prompt_feedback = getattr(response, "prompt_feedback", None)
                print(f"STEWARD_LLM_DIAG: text_len={text_len} candidates={cand_count} finish_reason={finish_reason}")
                error_classification = None
                if text_len == 0:
                    pf_safety = getattr(prompt_feedback, "safety_ratings", None) if prompt_feedback else None
                    pf_block = getattr(prompt_feedback, "block_reason", None) if prompt_feedback else None
                    citation_info = citation_metadata or citations
                    print(
                        f"STEWARD_LLM_EMPTY_RESPONSE: finish_reason={finish_reason} safety={safety_ratings} "
                        f"prompt_feedback={{'block_reason': {pf_block}, 'safety': {pf_safety}}} citations={citation_info} "
                        f"prompt_length_chars={len(system_prompt)}"
                    )
                    error_classification = "EMPTY"
                elif text_len < 50:
                    error_classification = "TOO_SHORT"
                trace = {
                    "model": self.model.model_name,
                    "response_text_len": text_len,
                    "prompt_text_len": len(system_prompt),
                    "candidates": cand_count,
                    "finish_reason": str(finish_reason),
                    "safety_ratings": str(safety_ratings),
                    "timestamp": datetime.utcnow().isoformat(),
                    "error_classification": error_classification,
                }
                try:
                    os.makedirs("data", exist_ok=True)
                    import json as _json
                    with open("data/steward_llm_trace.json", "w", encoding="utf-8") as f:
                        _json.dump(trace, f, indent=2)
                except Exception:
                    pass
            except Exception as diag_err:
                print(f"STEWARD_LLM_DIAG_WARNING: {diag_err}")
            if not summary or len(summary) < 10:
                # Fallback deterministic summary to avoid blank output
                shape = df.shape
                cols = [str(c) for c in df.columns[:20]]
                null_sample = df.isna().mean().round(3).to_dict()
                summary = (
                    f"DATA SUMMARY: Fallback deterministic summary. Rows={shape[0]}, Cols={shape[1]}, "
                    f"Columns={cols}. Null_frac_sample={null_sample}"
                )
            
            # Enforce Prefix
            if not summary.startswith("DATA SUMMARY:"):
                summary = "DATA SUMMARY:\n" + summary

            return {
                "summary": summary, 
                "encoding": detected_encoding,
                "sep": sep,
                "decimal": decimal,
                "file_size_bytes": file_size,
                "profile": dataset_profile,
            }
            
        except Exception as e:
            return {
                "summary": f"DATA SUMMARY: Critical Error analyzing data: {e}", 
                "encoding": detected_encoding,
                "sep": sep, 
                "decimal": decimal,
                "profile": {},
            }

    def _detect_csv_dialect(self, data_path: str, encoding: str) -> Dict[str, str]:
        """
        Robustly detects separator and decimal using csv.Sniffer and internal heuristics.
        """
        try:
            with open(data_path, 'r', encoding=encoding) as f:
                sample = f.read(50000) # 50KB sample
            
            # 1. Delimiter Detection
            try:
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample, delimiters=[',', ';', '\t', '|'])
                sep = dialect.delimiter
            except:
                # Fallback Heuristic
                if sample.count(';') > sample.count(','):
                    sep = ';'
                else:
                    sep = ','
            
            # 2. Decimal Detection
            decimal = self._detect_decimal(sample)
            diagnostics: Dict[str, Any] = {}
            if sep == decimal:
                diagnostics["ambiguous_sep_decimal"] = True
                if sep == ",":
                    alt_decimal = "."
                elif sep == ";":
                    alt_decimal = ","
                else:
                    alt_decimal = "," if sep == "." else "."
                diagnostics["decimal_candidates"] = [decimal, alt_decimal]
                diagnostics["selected_decimal"] = alt_decimal
                decimal = alt_decimal

            return {"sep": sep, "decimal": decimal, "diagnostics": diagnostics}
            
        except Exception as e:
            print(f"Steward: Dialect detection failed ({e}). Defaulting to standard.")
            return {"sep": ",", "decimal": "."}

    def _detect_decimal(self, text: str) -> str:
        """
        Analyzes numeric patterns to decide between '.' and ',' as decimal separator.
        """
        # Look for explicit float patterns: 123.45 vs 123,45
        dot_floats = re.findall(r'\d+\.\d+', text)
        comma_floats = re.findall(r'\d+,\d+', text)
        
        # We need to distinguish "comma as thousands sep" from "comma as decimal"
        # Heuristic: If we see many "123,45" but few "123.45", it's likely European.
        # However, "1,000" (thousands) vs "1,000" (small decimal) is hard.
        # Better simple check: 
        # If sep is ';', likely decimal is ','
        # If sep is ',', likely decimal is '.'
        
        # Let's count occurrences
        if len(comma_floats) > len(dot_floats) * 2:
            return ','
        
        return '.'

    def _smart_profile(self, df: pd.DataFrame, objective: str) -> Dict[str, str]:
        """
        Generates intelligent profile: High Card checks, Constant check, Target Detection.
        """
        alerts = ""
        col_details = ""
        ambiguities = ""
        glossary = ""
        ids = []
        dates = []
        targets = []
        
        # Keyword alignment
        obj_tokens = set(re.sub(r'[^a-z0-9]', ' ', objective.lower()).split())
        target_keywords = {'target', 'label', 'churn', 'class', 'outcome', 'y', 'status', 'revenue', 'sales'}
        target_keywords.update(obj_tokens)
        
        # Sort columns by importance (heuristic: keyword match -> numeric -> other)
        # We limit specific details to top 50
        all_cols = df.columns.tolist()
        priority_cols = []
        other_cols = []
        
        for col in all_cols:
            if any(k in col.lower() for k in target_keywords):
                priority_cols.append(col)
                if 'id' not in col.lower():
                    targets.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                priority_cols.append(col)
            else:
                other_cols.append(col)
                
        sorted_cols = (priority_cols + other_cols)[:50]

        def _norm_header(name: str) -> str:
            cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
            return re.sub(r"_+", "_", cleaned)

        name_collisions = {}
        for col in all_cols:
            normed = _norm_header(col)
            if normed:
                name_collisions.setdefault(normed, []).append(str(col))

        spaced_cols = [c for c in all_cols if " " in str(c) or "\t" in str(c)]
        trimmed_cols = [c for c in all_cols if str(c) != str(c).strip()]
        punct_cols = [c for c in all_cols if re.search(r"[^0-9A-Za-z_ ]", str(c))]
        collision_examples = [f"{k}: {v}" for k, v in name_collisions.items() if len(v) > 1]

        if trimmed_cols:
            sample = trimmed_cols[:5]
            ambiguities += f"- Column names have leading/trailing whitespace (e.g., {sample}); preserve exact names and account for whitespace when matching.\n"
        if spaced_cols:
            sample = spaced_cols[:5]
            ambiguities += f"- Column names contain spaces (e.g., {sample}); preserve exact names and use explicit mapping if needed.\n"
        if punct_cols:
            sample = punct_cols[:5]
            ambiguities += f"- Column names contain punctuation/special chars (e.g., {sample}); preserve exact names and map carefully.\n"
        if collision_examples:
            sample = collision_examples[:3]
            ambiguities += f"- Canonicalization collisions after normalization (examples: {sample}); disambiguate in mapping.\n"

        for col in sorted_cols:
            dtype = str(df[col].dtype)
            n_unique = df[col].nunique()
            from src.utils.missing import is_effectively_missing_series
            null_pct = is_effectively_missing_series(df[col]).mean()
            
            # Cardinality Check
            unique_ratio = n_unique / len(df) if len(df) > 0 else 0
            
            card_tag = ""
            if unique_ratio > 0.98 and n_unique > 50:
                card_tag = "[HIGH CARDINALITY/ID]"
                if 'id' not in col.lower():
                    ids.append(col)
            elif n_unique <= 1:
                card_tag = "[CONSTANT/USELESS]"
                alerts += f"- ALERT: '{col}' is constant (Value: {df[col].dropna().unique()}).\n"
            
            # Date Check (Robust)
            if df[col].dtype == 'object':
                 try:
                    # Sample Check for speed
                    sample_series = df[col].dropna().sample(min(len(df), 100), random_state=42)
                    parsed = pd.to_datetime(sample_series, errors='coerce', dayfirst=True)
                    if parsed.notna().mean() > 0.7:
                        dates.append(col)
                        card_tag += " [DATE-LIKE]"
                 except:
                    pass
            
            col_details += f"- {col}: {dtype}, Unique={n_unique} {card_tag}, Nulls={null_pct:.1%}\n"

            if dtype == "object":
                series = df[col].dropna().astype(str)
                if not series.empty:
                    sample = series.sample(min(len(series), 50), random_state=42)
                    percent_like = sample.str.contains("%").mean()
                    comma_decimal = sample.str.contains(r"\d+,\d+").mean()
                    dot_decimal = sample.str.contains(r"\d+\.\d+").mean()
                    numeric_like = sample.str.contains(r"^[\s\-\+]*[\d,.\s%]+$").mean()
                    whitespace = sample.str.contains(r"^\s+|\s+$").mean()
                    if numeric_like > 0.6:
                        ambiguities += f"- {col}: numeric-looking strings (~{numeric_like:.0%}); may need numeric conversion.\n"
                    if percent_like > 0.1:
                        ambiguities += f"- {col}: percent sign present (~{percent_like:.0%}); may need percent normalization.\n"
                    if comma_decimal > 0.1 and dot_decimal < 0.1:
                        ambiguities += f"- {col}: comma decimal pattern (~{comma_decimal:.0%}); likely decimal=','.\n"
                    if whitespace > 0.1:
                        ambiguities += f"- {col}: leading/trailing spaces (~{whitespace:.0%}); strip whitespace.\n"

            tokens = [t for t in col.lower().split("_") if t]
            role_hints = []
            if any(t in {"id", "uuid", "key"} for t in tokens):
                role_hints.append("identifier")
            if any(t in {"date", "fecha", "fec", "time"} for t in tokens):
                role_hints.append("date/time")
            if any(t in {"score", "risk", "rating"} for t in tokens):
                role_hints.append("score")
            if any(t in {"amount", "importe", "price", "cost", "monto"} for t in tokens):
                role_hints.append("monetary")
            if any(t in {"pct", "percent", "ratio", "rate"} for t in tokens):
                role_hints.append("percentage/ratio")
            if any(t in {"flag", "is", "has", "impacto", "status"} for t in tokens):
                role_hints.append("binary/flag")
            if role_hints:
                sample_vals = df[col].dropna().astype(str).head(3).tolist()
                glossary += f"- {col}: dtype={dtype}, hints={role_hints}, sample={sample_vals}\n"
            
        # Target Validation
        if targets:
            main_target = targets[0] # Best guess
            if df[main_target].nunique() <= 1:
                alerts += f"\n*** CRITICAL: Potential Target '{main_target}' has NO VARIATION. Modeling impossible. ***\n"

        # Representative Examples
        try:
            examples = df.sample(min(len(df), 3), random_state=42).to_string(index=False)
        except:
            examples = df.head(3).to_string(index=False)
            
        return {
            "column_details": col_details,
            "alerts": alerts,
            "ambiguities": ambiguities or "None detected.",
            "glossary": glossary or "None.",
            "ids": ids,
            "dates": dates,
            "targets": targets,
            "examples": examples
        }


def _infer_type_hint(series: pd.Series) -> str:
    dtype = str(series.dtype)
    if dtype.startswith("int") or dtype.startswith("float"):
        return "numeric"
    if dtype == "bool":
        return "boolean"
    if dtype.startswith("datetime"):
        return "datetime"
    if dtype == "object":
        sample = series.dropna().astype(str).head(200)
        if sample.empty:
            return "categorical"
        try:
            parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
            if parsed.notna().mean() > 0.7:
                return "datetime"
        except Exception:
            pass
        numeric_like = sample.str.contains(r"^[\s\-\+]*[\d,.\s%]+$").mean()
        if numeric_like > 0.7:
            return "numeric"
        return "categorical"
    return "unknown"


def detect_pii_findings(df: pd.DataFrame, threshold: float = 0.3) -> Dict[str, Any]:
    patterns = {
        "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", re.IGNORECASE),
        "PHONE": re.compile(r"(?:\+\d{1,3})?[-. (]*\d{3}[-. )]*\d{3}[-. ]*\d{4}", re.IGNORECASE),
        "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "IBAN": re.compile(r"[a-zA-Z]{2}\d{2}[a-zA-Z0-9]{4,}", re.IGNORECASE),
    }
    findings: List[Dict[str, Any]] = []
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        values = df[col].dropna().astype(str).tolist()
        if not values:
            continue
        sample = values[: min(len(values), 200)]
        for pii_type, pattern in patterns.items():
            match_count = sum(1 for val in sample if pattern.search(val))
            ratio = match_count / max(len(sample), 1)
            if ratio >= threshold:
                findings.append(
                    {
                        "column": col,
                        "pii_type": pii_type,
                        "match_ratio": round(ratio, 4),
                    }
                )
                break
    return {"detected": bool(findings), "findings": findings}


def build_dataset_profile(
    df: pd.DataFrame,
    objective: str,
    dialect_info: Dict[str, Any],
    encoding: str,
    file_size_bytes: int,
    was_sampled: bool,
    sample_size: int,
    pii_findings: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    columns = [str(c) for c in df.columns]
    type_hints = {col: _infer_type_hint(df[col]) for col in columns}
    missing_frac: Dict[str, float] = {}
    cardinality: Dict[str, Any] = {}
    suspected_ids: List[str] = []
    suspected_dates: List[str] = []
    suspected_targets: List[str] = []

    obj_tokens = set(re.sub(r"[^a-z0-9]", " ", (objective or "").lower()).split())
    target_keywords = {"target", "label", "churn", "class", "outcome", "y", "status", "revenue", "sales"}
    target_keywords.update(obj_tokens)

    from src.utils.missing import is_effectively_missing_series

    for col in columns:
        series = df[col]
        try:
            missing_frac[col] = float(is_effectively_missing_series(series).mean())
        except Exception:
            missing_frac[col] = float(series.isna().mean())
        n_unique = int(series.nunique(dropna=True))
        top_values = []
        try:
            counts = series.astype(str).value_counts(dropna=False).head(5)
            top_values = [{"value": str(idx), "count": int(cnt)} for idx, cnt in counts.items()]
        except Exception:
            top_values = []
        cardinality[col] = {"unique": n_unique, "top_values": top_values}
        if "id" in col.lower() or (len(df) > 0 and n_unique / max(len(df), 1) > 0.98):
            suspected_ids.append(col)
        if type_hints.get(col) == "datetime":
            suspected_dates.append(col)
        if any(tok in col.lower() for tok in target_keywords):
            suspected_targets.append(col)

    profile = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": columns,
        "type_hints": type_hints,
        "missing_frac": missing_frac,
        "cardinality": cardinality,
        "suspected_ids": sorted(set(suspected_ids)),
        "suspected_dates": sorted(set(suspected_dates)),
        "suspected_targets": sorted(set(suspected_targets)),
        "pii_findings": pii_findings or {"detected": False, "findings": []},
        "sampling": {
            "was_sampled": bool(was_sampled),
            "sample_size": int(sample_size),
            "file_size_bytes": int(file_size_bytes),
        },
        "dialect": {
            "sep": dialect_info.get("sep"),
            "decimal": dialect_info.get("decimal"),
            "encoding": encoding,
            "diagnostics": dialect_info.get("diagnostics") or {},
        },
    }
    return profile


def write_dataset_profile(profile: Dict[str, Any], path: str = "data/dataset_profile.json") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import json as _json
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(profile, f, indent=2, ensure_ascii=True)
    except Exception:
        return
