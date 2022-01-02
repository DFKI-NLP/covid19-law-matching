import json
from typing import List, Dict

from requests import get
from retry import retry

ANNOTATION_URL = "https://hypothes.is/api/search?user=niklas_thesis"


@retry(tries=3, delay=2)
def fetch_annotations() -> List[Dict]:

    offset = 0
    limit = 150
    annotations = []
    while True:
        r = get(ANNOTATION_URL + f"&offset={offset}&limit={limit}")
        rows = json.loads(r.text)["rows"]

        if len(rows) == 0:
            break

        annotations = annotations + rows
        offset += 150

    return annotations
