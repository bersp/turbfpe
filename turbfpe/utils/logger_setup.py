import logging

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s [%(levelname)s] [%(funcName)20s] %(message)s"),
)
