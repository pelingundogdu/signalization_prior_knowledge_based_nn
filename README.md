**DATASET**

<p><small>dataset link, https://drive.google.com/drive/folders/1qtCAPsPMznjMZ57BscAFgd-8Cnq9KnU3?usp=sharing

the data folder should be located outside of the project ../00_data/signalization_prior_knowledge_based_nn</small></p>

------------


**TRELLO**
<p><small>https://trello.com/b/ivgekKYB/draft</small></p>

------------


Project Organization
------------

    ├── environment.yml                <- The environment file
    │
    ├── LICENSE
    │
    ├── README.md                      <- The top-level README for developers using this project.
    │
    ├── ../00_data/{PROJECT_NAME}
    │   ├── external                   <- Data from third party sources.
    │   ├── interim                    <- Intermediate data that has been transformed.
    │   ├── processed                  <- The final, canonical data sets for modeling.
    │   └── raw                        <- The original, immutable data dump.
    │
    ├── models                         <- Trained and serialized models, model predictions, or model summaries
    │   ├── NN                         <- training and testing models
    │   └── CV                         <- cross-validation performance result
    │
    ├── notebooks                      <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                     the creator's initials, and a short `-` delimited description, e.g.
    │                                     `1.0-jqp-initial-data-exploration`.
    │
    ├── references                     <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                        <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── activation                 <- pathways activation results
    │   ├── CV                         <- the results of cross-validation performance
    │   ├── encoding                   <- the results of ecoding infomation
    │   ├── figures                    <- Generated graphics and figures to be used in reporting
    │   └── retrieval                  <- the results of retrieval performance
    │
    └── scripts
        ├── data-pbk                   <- prior biological knowledge from hipathia
        ├── dataset_scripts.py         <- scripts for dataset modificiation
        ├── model_scripts.py           <- scripts for getting tarined model
        ├── nn_desing_scripts.py       <- scripts for proposed NN
        ├── path_scripts.py            <- scripts for location 
        ├── settings.py                <- the main script file
        └── visualization_scripts.py   <- scripts for visualization
    

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
