

Building a model for a "gambler" who wants to make money betting.


Our goal is to:

1. Build a model for predicting match result
2. Instead of discrete categories we want to predict probabilities or ods

Results:
- simple classifier metrics like f1, accuracy

Result:
"Simulator" for testing betting strategies:
- Player starts at day Y, bets X amount of money on teams based on difference between ods predicting by model and offered by betting companies
- We evaluate won amount based on results + betting company ods
- We retrain the model with new results
- We run this for full season? Based on previous season results

Bonus points:
- Build league specific model
- Add weights to match results based on recency



Additional Features:
- [DONE] Rolling goal scored data
- [DONE] Season points and last season points
- [DONE] Sum of player overall rating
- [???] Team Elo Score (ratio or difference?)


Next:
- [???] Rolling Seasons point (?) e.g. last 10 days
- [???] Gap to 

Model to improve:
[???] Try SVM?
[???] Hyper Param tunning?
[???] League specific models

[   ] Include betting ods from betting companies (after all we can use them to predict games)

EDA to do:

- Draw data by team
  - We need to figure out why predicting draws is so hard
  - Included rolling draw ratio of each team

- Elo analysis:
 - Does it conform to expected distribution

- League analysis
  - Gini point distribution for all teams/seasons
  - Gaol scoring data etc.

Vaidation:
[???] Accuracy by league


EDA Visualizations:


[  ] For country/league confusion matrix instead show a interactive map of Europe with dropdown for each class etc.



TODO:

[  ] Look into Seaborn :The truth value of a <class 'cudf.core.series.Series'> is ambiguous: bug


[//]: # (https://www.kaggle.com/datasets/jiezi2004/soccer/data)