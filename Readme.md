## Links to HTML Notebooks/Presentations:

- [EDA](https://qwyt.github.io/m2s4_football_publish/EDA.html) \([with code](https://qwyt.github.io/m2s4_football_publish/EDA_with_code.html)\)
- [Model and Betting Strategy](https://qwyt.github.io/m2s4_football_publish/TabularModel.html) \([with code](https://qwyt.github.io/m2s4_football_publish/TabularModel_with_code.html)\)


This project is based on the [Ultimate 25k+ Matches Football Dataset](https://www.kaggle.com/datasets/prajitdatta/ultimate-25k-matches-football-database-european)

### Introduction and Goals

The dataset includes results of all football matches played in multiplem European leagues between including various other data related to player and teams (mainly based on the EA FIFA games). 

In addition to that it also includes the betting odds provided by various companies for each match.

**Our goal is to build a classification model which would allow use to predict the results of football matches and to attempt to create a profitable betting strategy using that model**

The project is split into 3 parts:
1. Exploratory analysis (`EDA.ipynb`), an overview of the teams and leagues, and individual player performance:
    1.1. Goal/score statistics and their variance between leagues
    1.2. Types of goals and variance between leagues
    1.3. Differences between players and their performance
    1.4. Inequality of team performance in different countries
    1.5. PCA and clustering of team performance and tactics/style attributes
    1.6. etc.
2. Model design, training and validation (`TabularModel.ipynb`):
    2.1. Feature and model selection
    2.2. Model hyperparameter tuning
    2.3. Performance cross-validation and using standard classification metrics, confusion matrices, AUC and evaluating probability prediction accuracy.
3. Designing a competitive betting strategy  (`TabularModel.ipynb`):
     We use the results from the previous section to select the model which is best at predicting the propabilities/odds of match outcomes rather than classification accuracy. We build a method which would allow us to select a subset of matches to bet on based on the predictive power of our model. On top of that we use various approaches to build an optimal betting strategy (e.g. using the Kelly criterion) to determine the optimal bet amount for each match.
    
    We were able to build a strategy which slightly outperforms the average betting odds offered by the companies included in the dataset over a selected subset of matches (around 5.5% return on investment over 75 games).


    3.1. Selecting the model best at predicting probabilities/odds rather than classification accuracy.

### Exploratory Analysis

### Feature Engineering

In addition to some of the variables directly available in the dataset we only use various more complex features derived from the dataset. Specifically:

- Rolling weighted (by time) and unweighted rolling data for match results, goals scored, points collected etc.
- Sum of all player ratings (based on EA FIFA games) playing in the match.


### Model

We've attempt to build XGBoost, Random Forest and more simple Logistic models. Reasonable classification performance (around 50% when predicting loss/draw/win) can be attained while including a very limited number of variables (home team advantage and sum of player ratings) both when using Logistic and more complex models. However when a much larger number of features are included the XGBoost model specifically significantly outperforms the Logistic model at predicting accurate probabilities, especially while using theresholds to select only matches with high confidence predictions (while Logistic models massively underperformed at probability predictions and made building a profitable betting strategy on top of them impossible) 

### Testing and Strategy Simulation


#### Main Challenges:

1. Player and Team attributes tables have multiple entries for each api_id because the attributes changed over time. When used any attribute stats in combination with the matches dataframe we have to always select the correct attr. row which applied at the given date


#### Additional technical details:
    - Most of the code code realted to data proccesing, feature engineering, graph drawing etc. is not included in the Jupyter notebooks, it can be found in the `workbench/src` folder:
        - *`model_config.py`* (includes moderl configurations, settings, hyperparameters, preprocessing, features included etc.)
        - `data_loader.py`
        - `data_process.py`
        - `graph.py`
        etc.

