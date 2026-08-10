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

After finding a common column, rows in both the tables have been manipulated such that every row in the match table has corresponding information in the points table. 

#### 

### Phase - 2: Feature Engineering:


### Phase - 3: Model fitting:


### Phase - 4: Summary and conclusion:






