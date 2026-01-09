## ML Ops Project
Our main objective with this project is to create a machine learning process that extracts data from Jobnet, transforms it, and uses it to train a model to match an external joblisting to one of Jobnet's predefined job categories.

The framework of the project is as follows:
1. **Data Extraction**: We extract joblisting data from Jobnet using (...). This data includes job titles, descriptions, and associated categories (**More?**).
2. **Data Transformation**: The extracted data is then cleaned and transformed to ensure it is in a suitable format for model training. This includes (...).
3. **Model Training**: We use the transformed data to train a machine learning model. Our baseline model is a sentence transformer that uses Cosine similarity as the similarity measure. Depending on how the data distributes into the predefined categories, we intend to use either a classification model (ANN) or a clustering approach (KNN) to categorize the joblistings.
4. **Model Evaluation**: After training, we evaluate the model's performance using appropriate metrics such as accuracy, precision, recall, and F1-score (**Hvis relevant**). We also perform cross-validation to ensure the model's robustness (**Igen, hvis relevant**).
5. **Deployment**: Finally, we deploy the trained model to a production environment where it can be used to classify new joblistings and match them to Jobnet's categories.