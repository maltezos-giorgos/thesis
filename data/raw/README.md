# Raw Datasets

This directory holds original, unmodified dataset files. It is excluded from git (see .gitignore).

## Required Datasets

### PopQA
- **Source:** https://github.com/AlexTMallen/adaptive-retrieval
- **Download:** `wget https://dl.fbaipublicfiles.com/dpr/data/retriever/popqa/popqa.csv`
- **Used for:** RQ1/RQ2/RQ3 evaluation; hard-negative mining

### TriviaQA
- **Source:** https://nlp.cs.washington.edu/triviaqa/
- **Download:** see `scripts/download_datasets.sh`
- **Used for:** Baseline reproduction (matches original CRAG paper)

### MS-MARCO Passages
- **Source:** https://microsoft.github.io/msmarco/
- **Used for:** Hard-negative mining pool

## Directory Layout After Download

```
data/raw/
├── popqa/
│   └── popqa.csv
├── triviaqa/
│   ├── triviaqa-rc.tar.gz
│   └── unzipped/
└── msmarco/
    └── collection.tsv
```

## Notes

- Do **not** commit any files from this directory.
- Run `make download-data` to invoke `scripts/download_datasets.sh` automatically.
- After downloading, run `scripts/reproduce_baseline.py` to verify dataset integrity.
