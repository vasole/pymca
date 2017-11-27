import logging
import traceback
_logger = logging.getLogger(__name__)

_logger.warning("%s is deprecated, you are advised to use "
                "silx.gui.plot.backends instead",
                __name__)
for line in traceback.format_stack(limit=4):
    _logger.warning(line.rstrip())
