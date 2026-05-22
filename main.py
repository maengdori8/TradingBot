import sys

from config.loader import load_config
from utils.logger import setup_logger


def main():
    config = load_config()
    setup_logger(config["logging"])

    run_mode = config["trading"].get("run_mode", "paper")

    if len(sys.argv) > 1:
        run_mode = sys.argv[1]

    if run_mode == "paper":
        from core.paper.engine import PaperEngine
        engine = PaperEngine()
    elif run_mode == "live":
        from core.engine import TradingEngine
        engine = TradingEngine()
    else:
        print(f"알 수 없는 모드: {run_mode} (paper / live)")
        sys.exit(1)

    engine.run()


if __name__ == "__main__":
    main()
