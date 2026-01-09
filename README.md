## ML Ops Project
Our main objective with this project is to create a machine learning process that extracts data from Jobnet, transforms it, and uses it to train a model to match an external joblisting to one of Jobnet's predefined job categories.

The framework of the project is as follows:
1. **Data Extraction**: We extract joblisting data from Jobnet using (...). This data includes job titles, descriptions, and associated categories (**More?**).
2. **Data Transformation**: The extracted data is then cleaned and transformed to ensure it is in a suitable format for model training. This includes (...).
3. **Model Training**: We use the transformed data to train a machine learning model. Our baseline model is a pretrained sentence transformer that uses Cosine similarity as the similarity measure. Depending on how the data distributes into the predefined categories, we intend to use either a classification model (ANN) or a clustering approach (KNN) to categorize the joblistings.
4. **Model Evaluation**: After training, we evaluate the model's performance using appropriate metrics such as accuracy, precision, recall, and F1-score (**Hvis relevant**). We also perform cross-validation to ensure the model's robustness (**Igen, hvis relevant**).
5. **Deployment**: Finally, we deploy the trained model to a production environment where it can be used to classify new joblistings and match them to Jobnet's categories.# mlopsg24

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
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
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
