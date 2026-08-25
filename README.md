# Momentum_Shifts_Tennis_Prediction
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

### Phase - 1: Exploratory data analysis

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
* 

Considering events, such as these, that signal a change in the way the game could be played out is an important characteristic of a momentum shift.

So the next step is to define a few features that can be seen as shifts in momentum and see how the points at which these shifts happen and how the point is won.

Feature 1 - Short term momentum:



Feature 2 - Long term momentum:





### Model fitting:


### Summary and conclusion:






