import sqlite3
import csv
from collections import Counter
from operator import itemgetter
from urllib.parse import urlsplit

import matplotlib.pyplot as plt
from texttable import Texttable
import latextable

con = sqlite3.connect("data/database.db")


def create_claim_stats(dataset_path="results/claim_extraction_12.09.21_0/dataset.csv"):
    with open(dataset_path, "r") as file:
        reader = csv.reader(file, delimiter=";")
        num_claims_list = []
        claims_length = []
        sample_length = []
        for sample in reader:
            list_of_claims = eval(sample[1])
            num_claims_list.append(len(list_of_claims))
            for start, end in list_of_claims:
                length = end - start
                claims_length.append(length)
            sample_length.append(len(sample[0]))

        num_samples = len(num_claims_list)
        fig, ax = plt.subplots()
        ax.hist(num_claims_list, max(num_claims_list))
        ax.set_xlabel("# claim per sample")
        ax.set_ylabel("# samples")
        plt.savefig("claims_per_sample.pdf")
        print(f"Total number of samples: {num_samples}")

        fig, ax = plt.subplots()
        ax.hist(claims_length, 10)
        ax.set_xlabel("Claim length (in chars)")
        ax.set_ylabel("# claims")
        plt.savefig("claims_length.pdf")

        fig, ax = plt.subplots()
        ax.hist(sample_length, 10)
        ax.set_xlabel("Sample length (in chars)")
        ax.set_ylabel("# samples")
        plt.savefig("sample_length.pdf")

        plt.show()


def create_matching_stats():
    res = con.execute('SELECT reference from "references"').fetchall()
    references = map(lambda t: t[0], res)
    number_of_references = list(map(lambda r: r.count("i.V.m.") + 1, references))
    breakpoint()
    fig, ax = plt.subplots()
    ax.hist(number_of_references, [0.5, 1.5, 2.5, 3.5, 4.5])
    ax.set_xlabel("# subsections per claim")
    ax.set_ylabel("# claims")
    ax.locator_params(axis="x", integer=True)
    plt.savefig("subsections_per_claim.pdf")


def create_sources_table():

    num_webpages = con.execute("SELECT COUNT(DISTINCT(url)) FROM fulltext").fetchone()[
        0
    ]
    all_urls = map(
        lambda x: x[0], con.execute("SELECT DISTINCT(url) FROM fulltext").fetchall()
    )
    base_urls = map(lambda url: urlsplit(url).netloc, all_urls)
    counts = Counter(base_urls)
    table_sources = Texttable()
    table_sources.set_cols_align(["l", "r"])
    table_sources.set_cols_valign(["t", "m"])
    table_sources.add_rows(
        [
            ["Source", "Num articles"],
            *[
                [k, v]
                for k, v in sorted(counts.items(), key=itemgetter(1), reverse=True)
            ],
            ["Total:", f"{num_webpages}"],
        ]
    )
    print("-- Sources: Basic --")
    print("Texttable Output:")
    print(table_sources.draw())
    print("\nLatextable Output:")
    print(
        latextable.draw_latex(
            table_sources, caption="Sources", label="table:sources_table"
        )
    )


if __name__ == "__main__":

    create_sources_table()
    create_claim_stats()
    create_matching_stats()
