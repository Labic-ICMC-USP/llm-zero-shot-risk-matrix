"""
Risk Assessor
-------------
Reads incidents from a CSV, uses an LLM to assign (1-5) severity and (1-5) likelihood,
plus a textual justification, and saves a new CSV with three extra columns.

With predict_likelihood: false in the config, the LLM is asked for severity and
justification only (no occurrence_likelihood column is written) -- for when
likelihood is produced by a separate method instead.

Requirements:
- pydantic>=2
- pyyaml
- pandas
- langchain_openai (already used by LLMExecutor)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Type
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import time
import random
import threading
import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------- Base executor (kept inline for self-contained project) -------
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from utils.source_config import load_source_config

ID_COLUMN = "report_id"  # used to detect already-scored rows when resuming an interrupted run


class LLMExecutor:
    def __init__(self):
        self.model_name = "mistralai/mistral-nemo"
        self.api_key: Optional[str] = None
        self.base_url: str = "https://openrouter.ai/api/v1"
        self.system_prompt: Optional[str] = None
        self.user_prompt: Optional[str] = None
        self.schema_class: Optional[Type[BaseModel]] = None
        self.schema_name = "response_schema"
        self.strict_mode = True

        # New parameters
        self.temperature: float = 0.0
        self.max_tokens: Optional[int] = None
        self.return_used_tokens = False
        self.reference_mode = False
        self.timeout: float = 60.0

    def set_model(self, model_name: str):
        self.model_name = model_name

    def set_api_key(self, api_key: str):
        self.api_key = api_key

    def set_base_url(self, base_url: str):
        self.base_url = base_url

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def set_user_prompt(self, prompt: str):
        self.user_prompt = prompt

    def set_schema(self, schema_class: Type[BaseModel], name: str = "response_schema", strict: bool = True):
        self.schema_class = schema_class
        self.schema_name = name
        self.strict_mode = strict

    def set_temperature(self, temperature: float):
        self.temperature = temperature

    def set_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens

    def set_return_used_tokens(self, enabled: bool):
        self.return_used_tokens = enabled

    def set_reference_mode(self, enabled: bool):
        self.reference_mode = enabled

    def set_timeout(self, timeout: float):
        self.timeout = timeout

    def run(self, user_prompt):
        if not self.api_key:
            raise ValueError("API key not configured.")
        if not self.schema_class:
            raise ValueError("Pydantic validation class not defined.")
        if not user_prompt:
            raise ValueError("User prompt not defined.")


        json_schema = self.schema_class.model_json_schema()
        json_schema.setdefault("additionalProperties", False)

        llm = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            timeout=self.timeout,
            model_kwargs={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.schema_name,
                        "strict": self.strict_mode,
                        "schema": json_schema
                    }
                }
            }
        )

        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=user_prompt))

        try:
            response = llm.invoke(messages)
            parsed_json = json.loads(response.content)

            if self.return_used_tokens:
                used_tokens = response.usage_metadata["total_tokens"]
                parsed_json["used_tokens"] = used_tokens

            result = self.schema_class(**parsed_json)
            return result

        except Exception as e:
            print(e)
            raise ValueError(f"Error processing LLM response: {e}")


# -------------------------------- Pydantic schemas --------------------------------

class RiskAssessment(BaseModel):
    """Structured output enforced by JSON Schema in the LLM response. Includes optional token usage info."""
    severity_level: int = Field(..., ge=1, le=4, description="Incident severity level from 1 (lowest) to 4 (highest), per the source's configured severity scale.")
    occurrence_likelihood: int = Field(..., ge=1, le=5, description="Likelihood of occurrence where 1=rare, 5=almost certain.")
    justification: str = Field(..., min_length=8, description="Short, evidence-based rationale for the chosen ratings.")
    used_tokens: Optional[int] = Field(None, description="Number of tokens used in this incident LLM evaluation.")
    scoring_failed: bool = Field(False, description="True when this row is a conservative fallback due to a scoring failure, not a real LLM assessment.")

    @field_validator("justification")
    @classmethod
    def must_be_explanatory(cls, v: str) -> str:
        if len(v.strip().split()) < 3:
            raise ValueError("Justification is too short; add a brief rationale.")
        return v.strip()


class SeverityAssessment(BaseModel):
    """Same as RiskAssessment minus occurrence_likelihood -- used when predict_likelihood
    is false, so the JSON Schema sent to the LLM doesn't require that field at all."""
    severity_level: int = Field(..., ge=1, le=4, description="Incident severity level from 1 (lowest) to 4 (highest), per the source's configured severity scale.")
    justification: str = Field(..., min_length=8, description="Short, evidence-based rationale for the chosen rating.")
    used_tokens: Optional[int] = Field(None, description="Number of tokens used in this incident LLM evaluation.")
    scoring_failed: bool = Field(False, description="True when this row is a conservative fallback due to a scoring failure, not a real LLM assessment.")

    @field_validator("justification")
    @classmethod
    def must_be_explanatory(cls, v: str) -> str:
        if len(v.strip().split()) < 3:
            raise ValueError("Justification is too short; add a brief rationale.")
        return v.strip()


class AppConfig(BaseModel):
    """YAML-driven configuration."""
    # LLM parameters
    api_key: str
    api_host: str = "https://openrouter.ai/api/v1"
    model_name: str = "mistralai/mistral-nemo"
    temperature: float = 0.0
    max_tokens: Optional[int] = None

    # Data parameters
    input_path: str
    output_path: str
    incident_text_column: str
    severity_level_column: Optional[str] = None  # required only when reference_mode is on

    # Optional behavior
    batch_size: int = 1  # sequential by default; adjust if you later add concurrency
    strict_schema: bool = True
    return_used_tokens: bool = False
    reference_mode: bool = False
    predict_likelihood: bool = True  # false = LLM scores severity + justification only

    # Retry behavior
    max_retries: int = 5
    retry_base_delay: float = 1.0  # seconds
    request_timeout: float = 60.0  # seconds -- prevents a hung API call from blocking forever

    # Which source config (config/sources/<source>.yaml) to pull
    # prompt_intro/severity_levels from -- prompt content is a property
    # of the data source, not of a single run, so it lives there
    # instead of being duplicated in every config.yaml.
    source: str

    @model_validator(mode="after")
    def _check_reference_mode_requirements(self) -> "AppConfig":
        if self.reference_mode and not self.severity_level_column:
            raise ValueError(
                "severity_level_column is required when reference_mode is true "
                "(it has no effect and can be omitted when reference_mode is false)."
            )
        return self

# ------------------------------- System & User prompts -------------------------------

# ------------------------------- System & User prompts -------------------------------
# The generic parts (role framing, response format, tone) are fixed here and shared by
# every source. Only the severity scale is domain-specific, so it's injected from the
# source config (prompt_intro/severity_levels in config/sources/<source>.yaml) instead
# of being hardcoded -- e.g. "Injury"/"Death" makes sense for saferproducts_gov but not
# for a financial-complaints source like cfpb.

def build_prompts(prompt_intro: str, severity_levels: dict, predict_likelihood: bool = True) -> tuple[str, str]:
    max_level = max(severity_levels.keys())
    levels_text = "\n".join(f"   {level} = {desc}" for level, desc in sorted(severity_levels.items()))

    # With predict_likelihood off, every likelihood line is dropped and the
    # justification step is renumbered; with it on, the text is unchanged.
    if predict_likelihood:
        likelihood_step = "3) Assign an OCCURRENCE LIKELIHOOD from 1 to 5 (1=rare, 2=unlikely, 3=possible, 4=likely, 5=almost certain).\n"
        justification_step = "4)"
        likelihood_json = '  "occurrence_likelihood": <int from 1 to 5>,\n'
        likelihood_reminder = "- occurrence_likelihood: integer 1–5 (1=rare, 5=almost certain)\n"
    else:
        likelihood_step = ""
        justification_step = "3)"
        likelihood_json = ""
        likelihood_reminder = ""

    system_prompt = f"""{prompt_intro}
Your job:
1) Read an incident description written in plain language.
2) Assign a SEVERITY LEVEL from 1 to {max_level}:
{levels_text}
{likelihood_step}{justification_step} Provide a short justification in one or two sentences, grounded in the description.
Return a STRICT JSON object following the provided JSON Schema; do not include extra fields or text.
Be concise, consistent, and avoid speculation not supported by the text.
"""

    user_prompt_template = f"""Incident description:
'''{{incident_text}}'''

Return a strict JSON object with exactly this structure:

{{{{
  "severity_level": <int from 1 to {max_level}>,
{likelihood_json}  "justification": "<short rationale, 1-2 sentences>"
}}}}

Remember:
- severity_level: integer 1–{max_level}
{levels_text}
{likelihood_reminder}- justification: short explanatory text
"""

    return system_prompt, user_prompt_template

# ------------------------------- Dataset severity level distribution -------------------------------
# Used when 'reference_mode' is true; computed per processed file (e.g. per category),
# not across the full multi-category dataset -- each category is treated as its own
# independent base for this reference distribution. Stored as instance state on
# RiskAssessor (see process_csv/score_incident) rather than as module globals, so
# multiple RiskAssessor instances don't share/overwrite each other's distribution.


# --------------------------------- RiskAssessor class ---------------------------------

class RiskAssessor:
    """Main service: loads config, runs LLM scoring over a CSV, and writes the result."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.executor = LLMExecutor()
        self.severity_distribution: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
        self.total_severity_count: int = 0

        source_config = load_source_config(self.config.source)
        prompt_intro = source_config["prompt_intro"]
        severity_levels = {int(k): v for k, v in source_config["severity_levels"].items()}
        self.system_prompt, self.user_prompt_template = build_prompts(
            prompt_intro, severity_levels, predict_likelihood=self.config.predict_likelihood
        )
        self.schema_class = RiskAssessment if self.config.predict_likelihood else SeverityAssessment

        self._configure_llm()

    def _fallback(self, justification: str):
        """Conservative lowest-risk default for a row that couldn't be scored."""
        extra = {"occurrence_likelihood": 1} if self.config.predict_likelihood else {}
        return self.schema_class(
            severity_level=1,
            justification=justification,
            scoring_failed=True,
            **extra,
        )

    def _configure_llm(self):
        self.executor.set_model(self.config.model_name)
        self.executor.set_api_key(self.config.api_key)
        self.executor.set_base_url(self.config.api_host)
        self.executor.set_system_prompt(self.system_prompt)
        self.executor.set_schema(self.schema_class, name="risk_assessment", strict=self.config.strict_schema)
        self.executor.set_temperature(self.config.temperature)
        self.executor.set_return_used_tokens(self.config.return_used_tokens)
        self.executor.set_reference_mode(self.config.reference_mode)
        self.executor.set_timeout(self.config.request_timeout)
        if self.config.max_tokens is not None:
            self.executor.set_max_tokens(self.config.max_tokens)

    def score_incident(self, text: str, severity_level: Optional[int] = None) -> RiskAssessment | SeverityAssessment:
        """Score a single incident description."""
        if not isinstance(text, str) or not text.strip():
            # Fallback when text is missing/invalid
            return self._fallback("Empty or invalid incident text; defaulting to lowest risk.")
        
        user_prompt = self.user_prompt_template.format(incident_text=text.strip())

        if self.config.reference_mode:
            ref_text = """Use the tips below to analyze:\n\nThe severity of this incident is: {severity_level}\nThe frequency with which this severity occurs is {severity_count} of {total_count}."""

            ref_text = ref_text.format(severity_level=severity_level, severity_count=self.severity_distribution[severity_level],
                                        total_count=self.total_severity_count)
            
            user_prompt += "\n\n" + ref_text

        max_retries = self.config.max_retries
        base_delay = self.config.retry_base_delay

        for attempt in range(max_retries):
            try:
                result = self.executor.run(user_prompt=user_prompt)  # returns self.schema_class
                if not isinstance(result, self.schema_class):
                    result = self.schema_class.model_validate(result)
                return result
            except Exception as e:
                # Se for a última tentativa, retorna default conservador
                if attempt == max_retries - 1:
                    return self._fallback(
                        f"Scoring failed after {max_retries} attempts ({type(e).__name__}); defaulting to lowest risk."
                    )
                # backoff exponencial + jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)

    def process_csv(
        self,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        reference_df: Optional[pd.DataFrame] = None,
    ) -> Path:
        """
        reference_df: optional full-category dataset used to compute the
        reference severity distribution when reference_mode is on. Use
        this when df (the rows actually being scored) is a partial
        subset (e.g. only missing or only failed rows) -- otherwise the
        reference distribution would be computed from that small subset
        instead of the full category. Defaults to the full input dataset.

        Resumable: if output_path already has rows (e.g. a previous run
        was interrupted), rows whose ID_COLUMN value is already present
        there are skipped, and new results are appended -- no manual
        merge/fix step needed after an interruption.
        """
        input_path = Path(input_path or self.config.input_path)
        output_path = Path(output_path or self.config.output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if input_path.suffix == ".parquet":
            full_df = pd.read_parquet(input_path)
        else:
            full_df = pd.read_csv(input_path)
        if self.config.incident_text_column not in full_df.columns:
            raise ValueError(
                f"Configured incident_text_column '{self.config.incident_text_column}' not found in CSV columns: {list(full_df.columns)}"
            )

        # Reference distribution is computed from the full dataset (or an
        # explicit reference_df) BEFORE any resume-filtering below, so it
        # never ends up reflecting only the leftover/unscored subset.
        if self.config.reference_mode:
            dist_source = reference_df if reference_df is not None else full_df
            if self.config.severity_level_column not in dist_source.columns:
                raise ValueError(
                    f"Configured severity_level_column '{self.config.severity_level_column}' not found in CSV columns: {list(dist_source.columns)}"
                )
            self.severity_distribution = dist_source[self.config.severity_level_column].astype(int).value_counts().to_dict()
            self.total_severity_count = dist_source.shape[0]

        df = full_df
        header_written = False
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            existing_df = pd.read_csv(output_path)
            if ID_COLUMN in existing_df.columns and ID_COLUMN in df.columns:
                already_scored = set(existing_df[ID_COLUMN])
                n_total = len(df)
                df = df[~df[ID_COLUMN].isin(already_scored)]
                print(f"Resuming: {len(df)} of {n_total} rows still need scoring "
                      f"({n_total - len(df)} already done in {output_path}).")
                header_written = True
            else:
                print(f"Existing output at {output_path} has no '{ID_COLUMN}' column to resume by -- starting fresh.")
                output_path.unlink()

        if df.empty:
            print("Nothing left to score.")
            return output_path

        if self.config.reference_mode:
            severities = df[self.config.severity_level_column].astype(int).tolist()
        else:
            severities = [None] * len(df)

        sev_col = "severity_level"
        lik_col = "occurrence_likelihood"
        just_col = "justification"
        failed_col = "scoring_failed"

        texts = df[self.config.incident_text_column].astype(str).tolist()

        write_lock = threading.Lock()
        n_failed = 0

        def write_row(idx: int, assessment: RiskAssessment | SeverityAssessment) -> None:
            nonlocal header_written
            row = df.iloc[[idx]].copy()
            row[sev_col] = assessment.severity_level
            if self.config.predict_likelihood:
                row[lik_col] = assessment.occurrence_likelihood
            row[just_col] = assessment.justification
            row[failed_col] = assessment.scoring_failed
            if self.executor.return_used_tokens:
                row["used_tokens"] = assessment.used_tokens
            with write_lock:
                row.to_csv(output_path, mode="a", header=not header_written, index=False)
                header_written = True

        pool = ThreadPoolExecutor(max_workers=self.config.batch_size)
        try:
            futures = {pool.submit(self.score_incident, t, s): idx for idx, (t, s) in enumerate(zip(texts, severities))}

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Scoring incidents"):
                idx = futures[fut]
                try:
                    assessment = fut.result()
                except Exception as e:
                    assessment = self._fallback(
                        f"Scoring failed ({type(e).__name__}); defaulting to lowest risk."
                    )
                if assessment.scoring_failed:
                    n_failed += 1
                write_row(idx, assessment)
        except KeyboardInterrupt:
            # Cancel rows that haven't started yet; the batch_size rows already
            # in flight still have to finish naturally (Python can't force-kill
            # a thread mid network call). Rows written so far are already saved
            # to output_path -- rerunning the same command later resumes from here.
            print(f"\nInterrupted -- cancelling queued rows, waiting on up to "
                  f"{self.config.batch_size} in-flight request(s) to finish...")
            pool.shutdown(wait=True, cancel_futures=True)
            print(f"Stopped. Progress saved to {output_path} -- rerun the same command to resume.")
            raise
        else:
            pool.shutdown(wait=True)

        if n_failed:
            print(f"Warning: {n_failed} of {len(df)} rows fell back to a conservative default (scoring_failed=True).")

        return output_path


# -------------------------------------- CLI helper --------------------------------------

def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)

def main():
    """
    Minimal CLI:
      python -m risk_scorer.risk_scorer config/config.yaml
    """
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m risk_scorer.risk_scorer <config.yaml>", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    assessor = RiskAssessor(cfg)
    out = assessor.process_csv()
    print(f"Done. Wrote: {out}")

if __name__ == "__main__":
    main()