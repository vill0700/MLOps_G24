## ML Ops Project
Our main objective with this project is to create a machine learning process that extracts data from Jobnet, transforms it, and uses it to train a model to match an external job vacancies to one of Jobnet's predefined job categories.

The framework of the project is as follows:
1. **Data Extraction**: From a private database we extract job vacancies from Jobnet. Data is tabular with fields of  the job vacancy text itself and ~20 classes of job type. Job vacancy texts contain personal information and noisy text not relevant to the job type, which is cleaned using GLINER2 configure to only extracts occupation, skills and task. These are then rewritten to a string to be more compatible with a standard SentenceTransformer embedding model. This data is made available to the entire group.
2. **Data Transformation**:
    The extracted data text are transformed into a text embedding using SentenceTransformer.
    Returns transformed data of labelled classes to vector embeddings of a pytorch tensor datatype of floats.
    The model chosen is *sentence-transformers/paraphrase-multilingual-mpnet-base-v2*, which is an older but standard multilingual text embedding model. For further performance could be considered, models such as *intfloat/multilingual-e5-large-instruct* can be used - which requires a prefix, or finetuning a text embedding model with SentenceTransformerTrainer using GenAI to judge triplet textdata of anchor: posive,negative data.
3. **Model Training**: We use the transformed data to train a machine learning model. Our baseline model is a pretrained sentence transformer that uses Cosine similarity as the similarity measure. We intend to add a classification model (ANN) trained on the data to improve performance.
4. **Model Evaluation**: After training, we evaluate the model's performance using appropriate metrics such as accuracy, precision, recall, and F1-score (**Hvis relevant**). We also perform cross-validation to ensure the model's robustness (**Igen, hvis relevant**).
5. **Deployment**: Finally, we deploy the trained model to a production environment where it can be used to classify new job vacancies and match them to Jobnet's categories.# mlopsg24

jobannonce classifier

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   ├── interrim
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── mlopsg24/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data_create.py    # extracts data from a private db - cannot be run
│   │   ├── data_preproces.py # module to embed texts
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
.
