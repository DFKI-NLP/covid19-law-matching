import typer

from scrape import scrape as scrape_function
from extract import extraction

app = typer.Typer()


@app.command()
def extract(
    prefix: str = typer.Option(None, help="Only extract those with prefix in name")
):
    extraction(prefix)


@app.command()
def scrape(
    url: str = typer.Option(None, help="Specify if a singe url should be scraped"),
    law: str = typer.Option(None, help="Name of the law"),
    file_with_urls: str = typer.Option(
        None, help="File in data/urls/{file} with urls to scrape"
    ),
):
    scrape_function(url, law, file_with_urls)


if __name__ == "__main__":
    app()
