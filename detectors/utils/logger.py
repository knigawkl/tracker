import logging
import sys

logger = logging.getLogger("Tracker")
logging.basicConfig(stream=sys.stdout, 
                    level=logging.INFO, 
                    format="%(asctime)s: %(message)s")
