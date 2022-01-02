import sqlite3
from datetime import datetime
from typing import List, Tuple

connection = sqlite3.connect("data/database.db")
cursor = connection.cursor()

today = datetime.today().date().strftime("%d.%m.%y")


def not_yet_processed(raw_annotation) -> bool:
    """Returns false if the annotations is already in the database."""
    cursor.execute(
        "SELECT * FROM claims WHERE annotation_id = ?", (raw_annotation["id"],)
    )
    return not bool(cursor.fetchone())


def fulltext_exists(url: str) -> bool:
    """Returns true if fulltext is already in database"""
    cursor.execute("SELECT * FROM fulltext WHERE url = ?", (url,))
    return bool(cursor.fetchone())


def _save_extraction_data(annotation, plaintext) -> None:
    cursor.execute(
        "INSERT INTO claims VALUES(?, ?, ?)",
        (annotation.id, annotation.claim, annotation.url),
    )
    if plaintext != "" and not fulltext_exists(annotation.url):
        cursor.execute(
            "INSERT INTO fulltext VALUES(?, ?, ?)",
            (annotation.url, plaintext, today),
        )


def _save_matching_data(annotation) -> None:
    cursor.execute(
        'INSERT INTO "references" VALUES(?, ?, ?)',
        (annotation.id, annotation.reference, annotation.date),
    )


def save_to_database(annotation, plaintext) -> None:
    if (
        not plaintext or plaintext.find(annotation.claim) == -1
    ) and not fulltext_exists(annotation.url):
        print(
            f"For url {annotation.url}: Claim not found in plaintext. Plaintext has to be manually added to fulltext"
            f"table."
        )
        plaintext = ""

    _save_extraction_data(annotation, plaintext)
    if annotation.for_law_matching and annotation.date:
        _save_matching_data(annotation)
    connection.commit()


def fetch_all_claims() -> List[Tuple]:
    return cursor.execute("SELECT * from claims").fetchall()


def fetch_all_fulltext() -> List[Tuple]:
    return cursor.execute("SELECT * from fulltext").fetchall()


def update_claim(claim, annotation_id):
    connection.execute(
        "UPDATE claims SET claim=? WHERE annotation_id=?", (claim, annotation_id)
    )
    connection.commit()


def update_fulltext(plaintext, url):
    connection.execute("UPDATE fulltext SET plaintext=? WHERE url=?", (plaintext, url))
    connection.commit()


def fetch_for_healthcheck():
    return connection.execute(
        "SELECT annotation_id, claim, plaintext FROM fulltext f join claims c on f.url=c.url"
    ).fetchall()


def fetch_for_claim_extraction_healthcheck():
    return connection.execute(
        "SELECT f.url, claim, plaintext FROM fulltext f join claims c on f.url=c.url"
    ).fetchall()
