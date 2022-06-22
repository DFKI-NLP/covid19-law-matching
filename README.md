# Claim Extraction and Law Matching for COVID-19-related Legislation

This is the repository for the paper [Claim retrieval and matching with laws for COVID-19 related legislation](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.50.pdf) (LREC 2022).

## Abstract

We examine the feasibility of a fact checking pipeline for claims about COVID-19-related laws in German, and define, implement, and benchmark two of its components: the Claim Extraction task and the Law Matching task. For both we provide hand-annotated data sets. In addition, we provide a data set of 13 COVID-19-related acts from Berlin, where each section is labeled with its applicability period.

We obtain promising results for both tasks with machine learning models based on the Transformer architecture, albeit with some conceptual limitations.

We also discuss challenges of machine learning in the legal domain, and show that complex legal reasoning tasks are insufficiently modeled.

## Table of Contents
[1. Setup](#setup)  
[2. Experiments](#experiments)  
[3. Training](#training)  
[4. Data sets](#data-sets)

## Setup

```console
git clone https://github.com/DFKI-NLP/covid19-law-matching.git paper
cd paper
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Experiments

The experiments from the thesis can be re-run via ``python training experiment [NR]``. All random seeds are set,
making the experiments reproducible.

| Experiment Nr. | Description                                               | Section in thesis |
|----------------|-----------------------------------------------------------|-------------------|
| 1              | Train & benchmark for Claim Extraction                    | 5.4.1             |
| 2              | Effect of different fold-composition for Claim Extraction | 5.4.2             |
| 3              | Visual results for Claim Extraction task                  | 5.4.3             |
| 4              | Train & benchmark for Law Matching task                   | 6.4.1             |

## Training

#### Claim extraction
````
Usage: training claim-extraction [OPTIONS]

Options:
  --epochs INTEGER                Number of epochs  [default: 3]
  --cross-validation / --no-cross-validation
                                  5-fold cross validation  [default: True]
  --inspect / --no-inspect        Sets breakpoint after model was trained, to
                                  interactively inspect results.  [default:
                                  False]

  --learning-rate FLOAT           [default: 2e-05]
  --model-checkpoint TEXT         [default: deepset/gbert-large]
  --help                          Show this message and exit.
````
e.g. ``python training claim-extraction --epochs 10``
#### Law matching
````
Usage: training law-matching [OPTIONS]

Options:
  --epochs INTEGER                Number of epochs  [default: 3]
  --cross-validation / --no-cross-validation
                                  5-fold cross validation  [default: True]
  --inspect / --no-inspect        Sets breakpoint after model was trained, to
                                  interactively inspect results.  [default:
                                  False]

  --learning-rate FLOAT           [default: 2e-05]
  --from-file TEXT                Load dataset from csv file with this path.
  --model-checkpoint TEXT         [default: deepset/gbert-large]
  --help                          Show this message and exit.
````
e.g. ``python training law-matching --epochs 10 --no-cross-validation --inspect``

## Data sets

We collected three data sets:
- Claim Extraction data set
- Law Matching data set
- COVID-19-related legislation in Berlin
### Claim Extraction data set

The data set for the Claim Extraction data set should be loaded from the database with the `ClaimExtractionDatasets` class:  
```python
from training.preprocessing.datasets_ import ClaimExtractionDatasets
datasets = ClaimExtractionDatasets.load_from_database()
assert len(datasets.train) == 63
assert len(datasets.test) == 16
assert len(list(datasets.folds)) == 5
datasets.save_to_csv('claim_extraction.csv') # for manual inspection
```

### Law Matching data set

```python
from training.preprocessing.datasets_ import LawMatchingDatasets
datasets = LawMatchingDatasets.load_from_database() # for new random negative samples
datasets = LawMatchingDatasets.load_from_csv('data/law_matching_dataset.csv') # for same samples as in experiments
assert len(datasets.train) == 686
assert len(datasets.test) == 172
assert len(list(datasets.folds)) == 5
datasets.save_to_csv('law_matching.csv') # for manual inspection, or to save data set
```

### Database

The SQLite database is `data/database.db`. It contains the unprocessed data for Claim Extraction and Law Matching data sets. For the data format
see [initial_migration.sql](data/initial_migration.sql) or `sqlite3 data/database.db -cmd .schema`

### Legislation

The legislation text can be found in the [legislation folder](data/legislation). The json files contain the section with their
respective validity dates. Currently, the following legislation is there:
- 1\. InfSchMV (from 16.12.2020 to 06.03.2021)
- 2\. InfSchMV (from 07.03.2021 to 17.06.2021)
- 3\. InfSchMV (from 18.06.2021 to 06.11.2021)
- Corona-ArbSchV (from 27.01.2021 to 10.09.2021)
- GroßveranstVerbV (from 22.04.2020 to 26.06.2020)
- SARS-CoV-2-EindV (from 14.03.2020 to 17.03.2020)
- Zweite Pflegemaßnahmen-Covid-19-Verordnung (from 25.02.2021 to 23.06.2021)
- 3\. PflegeM-Cov-19-V (from 24.06.2021 to 10.09.2021)
- Krankenhaus-Covid-19-Verordnung (from 17.10.2020 to 27.02.2021)
- Zweite Krankenhaus-Covid-19-Verordnung (from 25.02.2021 to 14.08.2021)
- SARS-CoV-2-Infektionsschutzverordnung (from 27.06.2020 to 15.12.2020)
- SchulHygCoV-19-VO (from 28.11.2020 to 08.08.2021)
- SARS-CoV-2-EindmaßnV (from 18.03.2020 to 26.06.2020)

The filename is `{abbreviation}.json`. The format is:
```json
{
  "name": "SARS-CoV-2-Arbeitsschutzverordnung",
  "abbreviation": "Corona-ArbSchV",
  "sections": [
    {
      "sectionNumber": "1",
      "sectionTitle": "Ziel und Anwendungsbereich",
      "valid_from": "27.01.2021",
      "valid_to": "12.03.2021",
      "text": "..."
    }
  ]
}
```

#### Scraping and parsing new legislation
For scraping and parsing new legislation from [gesetze.berlin.de](gesetze.berlin.de), their url should be placed in the `law_scraping/data/urls` folder.

####  Scrape:
```
Usage: law_scraping scrape [OPTIONS]

Options:
  --url TEXT             Specify if a singe url should be scraped
  --law TEXT             Name of the law
  --file-with-urls TEXT  File in data/urls/{file} with urls to scrape
  --help                 Show this message and exit.
```
e.g. `python law_scraping scrape --file-with-urls SchulHygCoV-19-VO.json`

#### Parse:
```
Usage: law_scraping extract [OPTIONS]

Options:
  --prefix TEXT  Only extract those with prefix in name
  --help         Show this message and exit.
```
e.g. `python law_scraping extract --prefix Schul`


## How to cite

Please cite [this paper](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.50.pdf) if you use our work:

```bibtex
@inproceedings{dehio2022,
  title = {{Claim Extraction and Law Matching for COVID-19-related
                  Legislation}},
  author = {Niklas Dehio and Malte Ostendorff and Georg Rehm},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Nicoletta Calzolari and Frédéric Béchet and Philippe Blache
                  and Christopher Cieri and Khalid Choukri and Thierry
                  Declerck and Hitoshi Isahara and Bente Maegaard and Joseph
                  Mariani and Jan Odijk and Stelios Piperidis},
  booktitle = {Proceedings of the 13th Language Resources and Evaluation
                  Conference (LREC~2022)},
  year = 2022,
  month = 6,
  address = {Marseille, France},
}
```

## License

MIT
