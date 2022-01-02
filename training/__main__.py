import typer

from train_claim_extraction_model import train_claim_extraction
from train_law_matching_model import train_law_matching
from baseline_law_matching import calculate_baseline_law_matching
from evaluate_law_matching import evaluate as _evaluate
from experiment1 import run_experiment1
from experiment2 import run_experiment2
from experiment3 import run_experiment3
from experiment4 import run_experiment4

app = typer.Typer()


@app.command()
def claim_extraction(
    epochs: int = typer.Option(3, help="Number of epochs"),
    cross_validation: bool = typer.Option(True, help="5-fold cross validation"),
    inspect: bool = typer.Option(
        False,
        help="Sets breakpoint after model was trained, to interactively inspect results.",
    ),
    learning_rate: float = 2e-5,
    model_checkpoint: str = typer.Option("deepset/gbert-large"),
):
    train_claim_extraction(
        epochs, cross_validation, inspect, learning_rate, model_checkpoint
    )


@app.command()
def law_matching(
    epochs: int = typer.Option(3, help="Number of epochs"),
    cross_validation: bool = typer.Option(True, help="5-fold cross validation"),
    inspect: bool = typer.Option(
        False,
        help="Sets breakpoint after model was trained, to interactively inspect results.",
    ),
    learning_rate: float = 2e-5,
    from_file: str = typer.Option(
        None, help="Load dataset from csv file with this path."
    ),
    model_checkpoint: str = typer.Option("deepset/gbert-large"),
):
    train_law_matching(
        epochs, cross_validation, inspect, learning_rate, from_file, model_checkpoint
    )


@app.command()
def experiment(nr: int):
    if nr == 1:
        run_experiment1()
        calculate_baseline_law_matching()
    elif nr == 2:
        run_experiment2()
    elif nr == 3:
        run_experiment3()
    elif nr == 4:
        run_experiment4()


@app.command()
def baseline_law_matching(
    from_file: str = typer.Option(
        None, help="Load dataset from csv file with this path."
    )
):
    calculate_baseline_law_matching(from_file)


@app.command()
def evaluate():
    _evaluate()


if __name__ == "__main__":
    app()
