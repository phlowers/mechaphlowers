# Logging for Mechaphlowers

## Logger inside

Mechaphlowers has a logger named `mechaphlowers` that is used by the library. It can be accessed from any module of the library.  
We propose to use it like this in a mechaphlowers module:
```python
import logging
logger = logging.getLogger(__name__)
# important to define the logger in this way, to avoid circular imports
# also in this way, the output can export the module name as well
# Output example: 
# 2025-03-28 21:33:47,634 - mechaphlowers.entities.arrays - DEBUG - Section Array initialized.

# later in the module code
logger.info("This is an info message")

```

## Logger outside

The mechaphlowers logger can be used from outside of the library.  
We propose to use it like this in an app using the library:  
```python
# In the main file, before importing mechaphlowers
import logging

Log = logging.getLogger("MyApp")
Log.setLevel(logging.DEBUG)
logging.basicConfig(filename='example.log', filemode="w", level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

Log.debug("This is a debug message")
# In the example.log file:
# 2025-03-28 21:33:41,771 - MyApp - DEBUG - This is a debug message

from mechaphlowers import SectionArray
# In the example.log file:
# 2025-03-28 21:33:42,437 - mechaphlowers - INFO - Mechaphlowers package initialized.
# 2025-03-28 21:33:42,437 - mechaphlowers - INFO - Mechaphlowers version: ...
```

## Logger in notebook

```python
import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.DEBUG, stream=sys.stdout)
logger.info('Logging is set up.')

# In the stdout:
# 2025-10-27 12:29:17,750 | INFO : Logging is set up.
# 2025-10-27 12:29:18,876 | INFO : Mechaphlowers package initialized.
# 2025-10-27 12:29:18,876 | INFO : Mechaphlowers version: 0.4.0b0
```

# Loger helper for debug uses

A helper is available to easily create a logger with the name of the current module for debug purposes:

::: mechaphlowers.utils.add_stderr_logger