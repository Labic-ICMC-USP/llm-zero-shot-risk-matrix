from pathlib import Path

from risk_scorer.risk_scorer import load_config, RiskAssessor

CONFIG_PATH = Path("config/config.yaml")


def run_risk_assessment() -> None:
    """
    Loads the dataset and LLM settings from config and runs the risk
    assessor, writing the scored CSV to the configured output path.
    """
    cfg = load_config(CONFIG_PATH)
    assessor = RiskAssessor(cfg)

    output_path = assessor.process_csv(cfg.input_path)
    print(f"Done. Wrote: {output_path}")


if __name__ == "__main__":
    run_risk_assessment()