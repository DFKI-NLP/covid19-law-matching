from requests import post
from retry import retry

WAYBACK_URL = "https://web.archive.org/save/"


@retry(tries=3, delay=2)
def save_to_wayback_machine(url_to_save: str) -> None:

    r = post(WAYBACK_URL + url_to_save)

    if not r.ok:
        raise Exception("Saving to Wayback Machine did fail.")
