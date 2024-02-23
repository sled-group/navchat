import logging
from datetime import datetime

logger = logging.getLogger(__name__)

logging.basicConfig(
    # filename=f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
    format="[%(asctime)s %(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
