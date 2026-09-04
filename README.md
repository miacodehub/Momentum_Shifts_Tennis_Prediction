# Momentum Shifts in Tennis and Predicting Winners
An ML project analyzing if wins can be predicted based on momentum shifts in matches.



## Problem statement:
Can we detect momentum shifts during a tennis match, and do those momentum shifts help predict the eventual match outcome?

## Data Source:
I used the data from Jeff Sackmann's match charting project. There's data for individual matches, individual points, and relevant statistics.
For this project, I only used match and point data from the 2020s.

## Exploratory Data Analysis

 
### Checking completeness of data:

The matches and points data has been checked for completeness and a final set(unique rows) of useful data has been prepared. The following are the details of this data:
Total matches: 3345
Total points: 547478
Date range: 2020-01-03 00:00:00 to 2026-05-21 00:00:00

The spread of matches over the years looks as follows:
<img width="1035" height="521" alt="image" src="https://github.com/user-attachments/assets/d19a61b3-3031-4080-89a8-674264df03ed" />

The distribution of matches will be lumpy as more data is charted for Grand Slams and famous players, less for smaller tournaments. 

### Checking if the tables are joined correctly 

There needs to be a column/feature that depicts the same thing in both the points and matches table. Only then, can it be said that the tables are joined correctly.

The matches table and points table are joined on match_id column. 

After finding a common column, rows in both the tables have been manipulated such that every row in the match table has corresponding information in the points table. If there are any matches for which there are no points, those rows have been dropped from the matches table. If there are any rows in the points table for which there is no match data, those rows have been dropped from the points table.

### Baseline numbers to keep track

Baseline numbers are the numbers a model has to beat in order to conclude that momentum can help predict point outcomes. One such baseline is the probability with which a server wins a point. This represents the structural advantage servers have in scoring, so it needs to be separated from any advantage due to momentum. In the current data, servers win 64% of points. The model has to beat that baseline to show that momentum adds real predictive value beyond serve advantage alone.

<img width="705" height="537" alt="image" src="https://github.com/user-attachments/assets/f9a1c4b5-432d-4481-8c8a-3ca2efc174c0" />

### Eyeball test on a single match

The purpose is to validate the data through human verification (an "eyeball test"). For this, I chose a single match and checked the following:

- Created a point_in_game column that tracks how many points have been played within the current game
- Checked that point_in_game resets to 1 every time a new game starts
- Checked that Gm1, Gm2, Set1, Set2 update correctly once a game or set is won
- Check that the last point number matches the sum of the final point in game of all games

<img width="607" height="382" alt="image" src="https://github.com/user-attachments/assets/dd28a6d5-3f05-4984-be4f-4736e8ff740b" />

<img width="632" height="376" alt="image" src="https://github.com/user-attachments/assets/e59b1cfc-4e78-48bc-8aba-173098f89ff3" />

### Validating match shape with what a real tennis match would look like

To confirm the dataset reflects real tennis rather than corrupted or mismatched data, 
I checked the distribution of points played per match. Professional matches typically 
run in a predictable range — roughly 100–250 points for a best-of-3 match, with longer 
5-set matches extending further. 

The initial distribution surfaced 6 matches with fewer than 40 points, well below what 
a completed match would produce — almost certainly retirements or walkovers rather than 
fully played matches. These were removed. After cleaning, the distribution centered 
around a mean of 164 points per match, with a realistic spread up to ~400 points for 
longer matches, and no remaining outliers below a plausible match length.

<img width="710" height="771" alt="image" src="https://github.com/user-attachments/assets/2f771c00-f7b1-49f2-b7bd-f1d544541238" />


## Feature Engineering:

Going back to the question being asked, what constitutes a momentum shift in tennis?

A momentum shift is defined as anything that breaks the flow of events in one way or another. Here are some examples:

* A streak of winning broken by a different player
* Server not winning the point (Servers have an advantage as they can set the pace of the game ith their serve so any time a server doesnt win a point, it's unusual and can be considered a shift in momentum.)
  

Considering events, such as these, that signal a change in the way the game could be played out is an important characteristic of a momentum shift.

So the next step is to define a few features that can be seen as shifts in momentum and see how the points at which these shifts happen and how the point is won.

Before creating features, I've changed the winning is represented.
Instead of keeping track of which player won, I'm keeping track of the first player.
If Player 1 won, point_p1 becomes point_p1 + 1. Else, point_p1 is point_p1 - 1;

### Feature 1 - Short term momentum (momentum_5):

This feature is to get an idea of who has been winning recently. I do that by taking the last 5 points and updating point_p1 as I go.
After the five points have been handled,  the feature momentum_5 will have a positive value, indicating that Player 1 is winning, or a negative value indicating that Player 2 is winning.

Note: The current point is not being considered in these calculations.


### Feature 2 - Long term momentum (momentum_10):

This feature is to get an idea of who has been winning in the longer term. I do that by taking the last 10 points and updating point_p1 as I go.
After the ten points have been handled,  the feature momentum_10 will have a positive value, indicating that Player 1 is winning, or a negative value indicating that Player 2 is winning.

This feature helps understand how the recent streak fares in comparison to a longer term picture.

For example:
If:
  momentum_5 = -1.0
  momentum_10 = -0.6
  It means, Player 2 has been especially dominating in the recent streak

if:
  momentum_5 = 1.0
  momentum_10 = -0.4
  It means, Player 1 has managed to turn the tide in his favor in the recent streak.

Note: The current point is not being considered in these calculations.

### Feature 3 - current streak (current_streak):

This feature shows how long the current player has been winning. If a different player wins, the value resets to 0 and starts increasing in the negative direction.

Just like the other two features, the variable is positive if Player 1 has been winning and negative if Player 2 has been winning.

This captures something that a rolling average doesn't necessarily capture: persistence.

### Feature 4 - Game score difference (game_score_diff):

This feature captures the current game advantage.

It is calculated as : game_score_diff = Gm1 - Gm2

If the value is :
+3 → P1 is ahead by 3 games
 0 → tied in games
-2 → P2 is ahead by 2 games

### Feature 5 - Set score difference (set_score_diff):

This is similar to game score difference but at the set level.

If the value is :
+1 → P1 has won one more set
 0 → sets are tied
-1 → P2 has won one more set

This gives the model larger-scale match context.

For example, momentum_5 = +1 doesn't indicate Player 1 is winning if Player 2 is ahead by 2 sets.

### Momentum Shift Feature 1 - momentum delta:

Momentum_delta = momentum_5 - momentum_10

This measures how much the recent momentum differs from the longer momentum.
A large positive difference means momentum has shifted towards Player 1.
A large negative difference means momentum has shifted towards Player 2.
A small difference or near 0 difference means there wasn't much of a shift in momentum.

This is a continuous shift signal.

### Momentum Shift Feature 2 - sign flip:

This flags when short term and long term momentum point in opposite directions.
This is a binary shift signal.

### Handling missing values:

The first few matches won't have enough information to calculate momentum_5 and momentum_10. 
Therefore, these values are removed from the data.

### Contextual Information:

The features 'Game Score Difference' and 'Set Score difference'are useful as contextual features.
These features show the condition of the games and the sets at the time of tracking this point.

## Modeling:

### Logistic Regression Model

####  Defining the Target:
The target in this model is the game winner as I'm modeling to see if I can predict the game winner.

game_winner = 1 → P1 eventually wins this game
game_winner = 0 → P2 eventually wins this game

#### Splitting the data:
I used a 80/20 rule for train/test split. So 80% of match points are used to train the model and 20% are used to test it.

#### Context-only Model:
Here, I built the model only using context a.k.a. using features game_score_diff and set_score_diff.
The following are the values I observed:

Accuracy : 56.53%
ROC-AUC : 0.541

#### Momentum-only Model:
Here, I built the model only using momentum features a.k.a. using features momentum_5, momentum_10, and current_streak.
The following are the values I observed:

Accuracy : 60.49%
ROC-AUC : 0.643

Compared with the context-only model, that's a pretty substantial jump.
Therefore, it can be concluded that the momentum model contains significantly more predictive information than a context-only model.

#### Full Model:
Here, I built the model with all 5 features.
The following are the values I observed:

Accuracy : 60.93%
ROC-AUC : 0.649

#### Momentum shift feature:

Until now, I've focused on building features that understand momentum and then predict the winner based on that understanding.
But the broader question is about momentum _shifts_ more than just momentum. 

The full model achieved 60.93% accuracy and an ROC-AUC of 0.649, indicating that the engineered momentum and contextual features contain meaningful predictive information about game outcomes

However, momentum shift is a bit different. It is used to indicate what might not be obvious - who's in the lead and what just happened that may indicate that the winner probably be the one in lead.



| Model                      |   Accuracy |   ROC-AUC |
| -------------------------- | ---------: | --------: |
| Context-only               |     56.53% |     0.541 |
| Momentum-only              |     60.49% |     0.643 |
| Full                       |     60.93% |     0.649 |
| **Full + momentum shifts** | **61.29%** | **0.655** |


So adding momentum_delta + sign_flip improved:

Accuracy: +0.36 percentage points
ROC-AUC: +0.0061

### XG Model:

Logistic Regression assumes a relatively simple relationship between the features and the outcome. Since the relationship between momentum, game context, and the eventual game winner may be more complex, I also tested an XGBoost model.

XGBoost (Extreme Gradient Boosting) is a tree-based model that can capture non-linear relationships and interactions between features.

#### Momentum-only XGBoost:

I first trained XGBoost using only the three momentum features:

* momentum_5
* momentum_10
* current_streak

The following are the values I observed:

Accuracy : 66.04%
ROC-AUC : 0.713
Log Loss : 0.622

This performed substantially better than the Logistic Regression momentum-only model, which achieved 60.49% accuracy and an ROC-AUC of 0.643.

#### Full XGBoost Model:

I then trained XGBoost using all five features:

* momentum_5
* momentum_10
* current_streak
* game_score_diff
* set_score_diff

The following are the values I observed:

Accuracy : 69.16%
ROC-AUC : 0.756
Log Loss : 0.588

Adding the game and set context improved the model compared with the momentum-only XGBoost model.

#### XGBoost with Momentum Shifts:

Finally, I added the two explicitly engineered momentum shift features:

* momentum_delta
* sign_flip

The following are the values I observed:

Accuracy : 68.97%
ROC-AUC : 0.755
Log Loss : 0.589

The addition of the momentum shift features did not improve the XGBoost model. The full XGBoost model without these features performed slightly better.

## Observations:

The XGBoost models performed substantially better than the Logistic Regression models.

The momentum-only XGBoost model achieved an ROC-AUC of 0.713, showing that the momentum features contain meaningful predictive information on their own.

Adding game and set context improved the ROC-AUC further to 0.756 and increased accuracy to 69.16%.

Interestingly, explicitly adding `momentum_delta` and `sign_flip` did not improve the XGBoost model. This suggests that XGBoost may already be able to capture the useful relationships between the underlying momentum features without requiring these additional shift features.



## Evaluation:

I evaluated the models using three metrics: Accuracy, ROC-AUC, and Log Loss.

**Accuracy** measures the percentage of game outcomes that the model predicted correctly.

**ROC-AUC (Receiver Operating Characteristic – Area Under the Curve)** measures how well the model separates games eventually won by Player 1 from games eventually won by Player 2. An AUC of 0.5 represents random classification, while a value closer to 1 indicates better separation.

**Log Loss** evaluates the quality of the predicted probabilities. Lower values indicate better-calibrated predictions, while confident incorrect predictions are penalized more heavily.

The final model results were:

| Model                             |   Accuracy |   ROC-AUC |  Log Loss |
| --------------------------------- | ---------: | --------: | --------: |
| Context-only                      |     56.53% |     0.541 |         — |
| Logistic - Momentum               |     60.49% |     0.643 |         — |
| Logistic - Full                   |     60.93% |     0.649 |         — |
| Logistic - Full + Momentum Shifts |     61.29% |     0.655 |         — |
| XGBoost - Momentum                |     66.04% |     0.713 |     0.622 |
| **XGBoost - Full**                | **69.16%** | **0.756** | **0.588** |
| XGBoost - Full + Momentum Shifts  |     68.97% |     0.755 |     0.589 |

The XGBoost Full model produced the strongest overall results, achieving 69.16% accuracy and an ROC-AUC of 0.756.

## Results and findings:

The results show that point-level momentum contains meaningful predictive information about the eventual winner of the current game.

The momentum-only Logistic Regression model improved from the context-only baseline of 56.53% accuracy and 0.541 ROC-AUC to 60.49% accuracy and 0.643 ROC-AUC.

Using XGBoost produced a much larger improvement. The momentum-only XGBoost model achieved 66.04% accuracy and 0.713 ROC-AUC.

Adding game and set context further improved the XGBoost model to 69.16% accuracy and 0.756 ROC-AUC, making it the best-performing model in the experiment.

The explicitly engineered momentum shift features (`momentum_delta` and `sign_flip`) produced a small improvement when added to Logistic Regression, increasing accuracy from 60.93% to 61.29% and ROC-AUC from 0.649 to 0.655.

However, these features did not improve XGBoost. The XGBoost model with the shift features achieved 68.97% accuracy and 0.755 ROC-AUC, compared with 69.16% accuracy and 0.756 ROC-AUC for the full model without them.

This suggests that while momentum shifts can provide some additional information to a simpler linear model, XGBoost is already able to capture useful relationships between the underlying momentum features.

## Limitations:

* **Momentum is an engineered concept:** Momentum is not directly observable in the data. The features used in this project are proxies based on recent point-winning patterns and therefore represent one possible definition of momentum.

* **Game-level target:** The model predicts the eventual winner of the current game, rather than the winner of the entire match.

* **Repeated observations within games:** Multiple point-level observations from the same game have the same eventual game winner as the target. Although the train/test split was performed at the match level to prevent points from the same match appearing in both sets, the individual point observations are not completely independent.

* **Serve advantage:** Serving provides a structural advantage in tennis. The dataset showed that servers won approximately 64% of points. The models focus on momentum and score context rather than explicitly modeling every aspect of serve-related advantage.

* **Data coverage:** The Match Charting Project does not contain an equally distributed sample of all professional tennis matches. Some tournaments and players are represented more heavily than others.

* **Predictive, not causal:** A relationship between momentum features and game outcomes does not establish that momentum itself causes a player to win.

* **Model generalization:** The models were evaluated using a match-level train/test split rather than a chronological split. Therefore, the results do not directly measure how well the model would perform when predicting games from completely future tournaments or seasons.

## Summary and conclusion:

This project explored whether point-level momentum patterns and momentum shifts can help predict the eventual winner of a tennis game.

The results indicate that momentum contains meaningful predictive information. Both Logistic Regression and XGBoost performed better when momentum features were included, with XGBoost achieving the strongest performance.

The best model was the **XGBoost Full model**, which achieved **69.16% accuracy, 0.756 ROC-AUC, and 0.588 Log Loss** using momentum and game/set context.

Explicit momentum-shift features provided a small improvement for Logistic Regression but did not improve XGBoost. This suggests that the underlying momentum features contain useful information, while explicitly encoding the shift may not add much once a nonlinear model is able to learn interactions between the features.

Overall, the project provides evidence that recent point-level patterns can be useful for predicting the eventual winner of the current tennis game, while also showing that the way momentum is represented and modeled has a significant impact on predictive performance.





