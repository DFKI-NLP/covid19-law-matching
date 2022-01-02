CREATE TABLE IF NOT EXISTS claims (
    annotation_id TEXT PRIMARY KEY,
    claim TEXT NOT NULL,
    url TEXT NOT NULL,
    FOREIGN KEY (url)
        REFERENCES fulltext (url)
);

CREATE TABLE IF NOT EXISTS fulltext (
    url TEXT NOT NULL,
    plaintext TEXT NOT NULL,
    saved_to_wayback_machine TEXT
);

CREATE TABLE IF NOT EXISTS "references" (
    annotation_id TEXT NOT NULL,
    reference TEXT NOT NULL,
    date TEXT NOT NULL,
    FOREIGN KEY (annotation_id) REFERENCES claims (annotation_id)
);
