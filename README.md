# Predicting Winner in Tennis based on Momentum
An ML project analyzing if wins can be predicted based on momentum shifts in matches.



## Problem statement:
Can we detect momentum shifts during a tennis match, and do those momentum shifts help predict the eventual match outcome?
I used the data from Jeff Sackmann's match charting project. There's data for individual matches, individual points, and relevant statistics.

## Tennis Theory:

### Tennis Terms:

- Break point: Returner (the player returning the serve) is 1 point from winning the game
- Trailing a set: 

### Tennis Momentum Model:
There is a long existing model in tennis called the Tennis Momentum Model. This model's goal is to explore how momentum and shifts in momentum affect player performance and eventually player match outcomes. 



## ML engineering

### Exploratory data analysis

For this project, I only used match and point data from the 2020s. 

#### Checking completeness of data:

The matches and points data has been checked for completeness and a final set(unique rows) of useful data has been prepared. The following are the details of this data:
Total matches: 3345
Total points: 547478
Date range: 2020-01-03 00:00:00 to 2026-05-21 00:00:00

The spread of matches over the years looks as follows:
<img width="1035" height="521" alt="image" src="https://github.com/user-attachments/assets/d19a61b3-3031-4080-89a8-674264df03ed" />

The distribution of matches will be lumpy as more data is charted for Grand Slams and famous players, less for smaller tournaments. 

#### Checking if the tables are joined correctly 

There needs to be a column/feature that depicts the same thing in both the points and matches table. Only then, can it be said that the tables are joined correctly.

The matches table and points table are joined on match_id column. 

After finding a common column, rows in both the tables have been manipulated such that every row in the match table has corresponding information in the points table. If there are any matches for which there are no points, those rows have been dropped from the matches table. If there are any rows in the points table for which there is no match data, those rows have been dropped from the points table.

#### Baseline numbers to keep track

Baseline numbers are the numbers a model has to beat in order to conclude that momentum can help predict point outcomes. One such baseline is the probability with which a server wins a point. This represents the structural advantage servers have in scoring, so it needs to be separated from any advantage due to momentum. In the current data, servers win 64% of points. The model has to beat that baseline to show that momentum adds real predictive value beyond serve advantage alone.

<img width="705" height="537" alt="image" src="https://github.com/user-attachments/assets/f9a1c4b5-432d-4481-8c8a-3ca2efc174c0" />

#### Eyeball test on a single match

The purpose is to validate the data through human verification (an "eyeball test"). For this, I chose a single match and checked the following:

- Created a point_in_game column that tracks how many points have been played within the current game
- Checked that point_in_game resets to 1 every time a new game starts
- Checked that Gm1, Gm2, Set1, Set2 update correctly once a game or set is won
- Check that the last point number matches the sum of the final point in game of all games

<img width="607" height="382" alt="image" src="https://github.com/user-attachments/assets/dd28a6d5-3f05-4984-be4f-4736e8ff740b" />

<img width="632" height="376" alt="image" src="https://github.com/user-attachments/assets/e59b1cfc-4e78-48bc-8aba-173098f89ff3" />

#### Validating match shape with what a real tennis match would look like

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


### Feature Engineering:

Going back to the question being asked, what constitutes a momentum shift in tennis?

A momentum shift is defined as anything that breaks the flow of events in one way or another. Here are some examples:

* A streak of winning broken by a different player
* Server not winning the point (Servers have an advantage as they can set the pace of the game ith their serve so any time a server doesnt win a point, it's unusual and can be considered a shift in momentum.)
  

Considering events, such as these, that signal a change in the way the game could be played out is an important characteristic of a momentum shift.

So the next step is to define a few features that can be seen as shifts in momentum and see how the points at which these shifts happen and how the point is won.

Before creating features, I've changed the winning is represented.
Instead of keeping track of which player won, I'm keeping track of the first player.
If Player 1 won, point_p1 becomes point_p1 + 1. Else, point_p1 is point_p1 - 1;

#### Feature 1 - Short term momentum (momentum_5):

This feature is to get an idea of who has been winning recently. I do that by taking the last 5 points and updating point_p1 as I go.
After the five points have been handled,  the feature momentum_5 will have a positive value, indicating that Player 1 is winning, or a negative value indicating that Player 2 is winning.

Note: The current point is not being considered in these calculations.


#### Feature 2 - Long term momentum (momentum_10):

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

#### Feature 3 - current streak (current_streak):

This feature shows how long the current player has been winning. If a different player wins, the value resets to 0 and starts increasing in the negative direction.

Just like the other two features, the variable is positive if Player 1 has been winning and negative if Player 2 has been winning.

This captures something that a rolling average doesn't necessarily capture: persistence.

#### Feature 4 - Game score difference (game_score_diff):

This feature captures the current game advantage.

It is calculated as : game_score_diff = Gm1 - Gm2

If the value is :
+3 → P1 is ahead by 3 games
 0 → tied in games
-2 → P2 is ahead by 2 games

#### Feature 5 - Set score difference (set_score_diff):

This is similar to game score difference but at the set level.

If the value is :
+1 → P1 has won one more set
 0 → sets are tied
-1 → P2 has won one more set

This gives the model larger-scale match context.

For example, momentum_5 = +1 doesn't indicate Player 1 is winning if Player 2 is ahead by 2 sets.


### Handling missing values:

The first few matches won't have enough information to calculate momentum_5 and momentum_10. 
Therefore, these values are removed from the data.

### Contextual Information:

The features 'Game Score Difference' and 'Set Score difference'are useful as contextual features.
These features show the condition of the games and the sets at the time of tracking this point.

### Modeling:

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


### Observations:



### Evaluation:

### Momentum Shift Analysis:

### Results and findings:

### Limitations:

### Summary and conclusion:






