"""Atomic single-writer JSON progress reports, including Windows reader locks."""
import json
import os
from pathlib import Path
import time

_WINDOWS = os.name == "nt"
_REPLACE_ATTEMPTS = 25


def write_json_report(path, document):
    """Never expose partial JSON; retry only a brief Windows rename lock.

    A Windows reader without FILE_SHARE_DELETE can block replace while it reads.
    The previous complete report stays readable until replacement succeeds. A
    persistent permission failure still raises after at most 0.48s of waits;
    this never retries an environment transition, search or training seed.
    Callers must ensure one writer per report (parent waits for child exit).
    """
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            if (not _WINDOWS or getattr(error, "winerror", None) not in {5, 32, 33}
                    or attempt + 1 == _REPLACE_ATTEMPTS):
                raise
            time.sleep(.02)
