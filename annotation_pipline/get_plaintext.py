from newsplease import NewsPlease
from retry import retry


@retry(tries=3, delay=2)
def fetch_plaintext(url: str) -> str:

    article = NewsPlease.from_url(url)

    return article.maintext or ""
