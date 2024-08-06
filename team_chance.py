import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
df = pd.read_csv("archive\\2012-18_teamBoxScore.csv")

#print(df.head)
df['gmDate'] = pd.to_datetime(df['gmDate'])
df['year'] = df['gmDate'].dt.year

nba_teams = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BRK': 'Brooklyn Nets', 'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers', 'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons', 'GS': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'Los Angeles Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies', 'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves', 'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder', 'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHO': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs', 'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
}

### PREDICTIVE MODELING
'''
Game Outcome Prediction: Use features such as team statistics, opponent statistics, and game conditions to predict the outcome of a game (win/loss).
'''
# Returns the data averages for a certain team within a range of years and whether the team won or lost
def data_by_game(team_name, years):
    stats = []
    result = [] # 1 is win, 0 is loss
    for index, row in df.iterrows():
        if (row["year"] in years and team_name == row["teamAbbr"]):
            game_record = [
                row["teamPTS"],
                row["teamAST"],
                row["teamTO"],
                row["teamSTL"],
                row["teamBLK"],
                row["teamPF"],
                row["teamFGA"],
                row["teamFGM"],
                row["teamFG%"],
                row["team2PA"],
                row["team2PM"],
                row["team2P%"],
                row["team3PA"],
                row["team3PM"],
                row["team3P%"],
                row["teamFTA"],
                row["teamFTM"],
                row["teamFT%"],
                row["teamORB"],
                row["teamDRB"],
                row["teamTRB"],
                row["teamOrtg"],
                row["teamDrtg"],
            
            ] + [1 if row["opptAbbr"] == other_team else 0 for other_team in nba_teams.keys()]
    
            stats.append(game_record)
            if row["teamRslt"] == "Win":
                result.append(1)
            else:
                result.append(0)
    return stats, result
stats, result = data_by_game("GS", [i for i in range(2012, 2019)])

#------------------LOGISTIC REGRESSION-----------------------------------------------------------
# Uses logistic regression to classify games as either "won" or "loss" based on 24 variables

def LogRegression(X_whole, y_whole, test_size, random_state, iterations):
    X_train, X_test, y_train, y_test = train_test_split(X_whole, y_whole, test_size=test_size, random_state = random_state)

    logreg = LogisticRegression(solver='lbfgs', max_iter=iterations, random_state=random_state)

    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    target_names = ["1", "0"]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # A sure victory
    test_statistics = [[150, 100, 0, 100, 100, 100, 50, 50, 1, 100, 100, 1, 10, 10, 1, 100, 100, 1, 50, 50, 100, 200, 200, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0, 0]]
    score = logreg.predict(test_statistics)
    print(score)

    # A sure loss
    test_statistics = [[30, 2, 50, 1, 2, 3, 25, 2, 0.04, 10 , 3, .3, 3, 2, .67, 3, 0, 0, 2, 2, 100, 34, 65, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0, 0]]
    score = logreg.predict(test_statistics)
    print(score)

    return logreg
#98.5% accuracy
#LogRegression(stats, result, 0.25, 16, 2000)
#-----------------------------------------------------------------------------

#------------------Decision Tree and Adaboost---------------------------------------------------------
def DecisionTree(X_whole, y_whole, test_size, random_state):
    features = [
    'teamPTS', 'teamAST', 'teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFGA',
    'teamFGM', 'teamFG%', 'team2PA', 'team2PM', 'team2P%', 'team3PA', 'team3PM',
    'team3P%', 'teamFTA', 'teamFTM', 'teamFT%', 'teamORB', 'teamDRB', 'teamTRB',
    'teamOrtg', 'teamDrtg',
    'ATL', 'BOS', 'BRK', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GS', 'HOU',
    'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL',
    'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
    ]
    dtree = DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X_whole, y_whole, test_size=test_size, random_state = random_state)
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X_train, y_train)
    tree.plot_tree(dtree, feature_names = features)
    plt.show()

    dtree.predict(X_test)
    score = dtree.score(X_test, y_test)
    print("Decision Tree Score:" , score)

    abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)
    model = abc.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Adaboost Accuracy:",metrics.accuracy_score(y_test, y_pred))
# 95.12% accuracy boosted to 96.74% with Adaboost
# DecisionTree(stats, result, 0.25, 16)


'''
Score Prediction: Predict the total score using regression algorithms.
''' 
# Returns a list of statistics and a list of scores corresponding to each statistic. 
def data_by_game(team_name, years):
    stats = []
    result = [] # scores
    for index, row in df.iterrows():
        if (row["year"] in years and team_name == row["teamAbbr"]):
            game_record = [
                row["teamAST"],
                row["teamTO"],
                row["teamSTL"],
                row["teamBLK"],
                row["teamPF"],
                row["teamFGA"],
                row["teamFGM"],
                row["teamFG%"],
                row["team2PA"],
                row["team2PM"],
                row["team2P%"],
                row["team3PA"],
                row["team3PM"],
                row["team3P%"],
                row["teamFTA"],
                row["teamFTM"],
                row["teamFT%"],
                row["teamORB"],
                row["teamDRB"],
                row["teamTRB"],
                row["teamOrtg"],
                row["teamDrtg"],
            ] + [1 if row["opptAbbr"] == other_team else 0 for other_team in nba_teams.keys()]
    
            stats.append(game_record)
            result.append(row["teamPTS"])
    return stats, result
stats, result = data_by_game('GS', [2012, 2013, 2014, 2015, 2016])

#------------------Linear Regression---------------------------------------------------------
def LinRegression(X_whole, y_whole, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X_whole, y_whole, test_size=test_size, random_state = random_state)
    linreg = LinearRegression()
    linreg = linreg.fit(X_train, y_train)
    coef = linreg.coef_
    intercept = linreg.intercept_
    y_pred = linreg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean squared error: ", mse)
    r2 = r2_score(y_test, y_pred)
    print("Coefficient of determination: %.2f" % r2)
    # High Scoring game
    test_statistics = [[100, 0, 100, 100, 100, 50, 50, 1, 100, 100, 1, 10, 10, 1, 100, 100, 1, 50, 50, 100, 200, 200, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0, 0]]
    score = linreg.predict(test_statistics)
    print(score)
    # Low Scoring Game
    test_statistics = [[2, 50, 1, 2, 3, 25, 2, 0.04, 10 , 3, .3, 3, 2, .67, 3, 0, 0, 2, 2, 100, 34, 65, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0, 0]]
    score = linreg.predict(test_statistics)
    print(score)
    
#LinRegression(stats, result, 0.25, 16)
# RMSE: 1.5712
#------------------XGBoost---------------------------------------------------------
def XGBoost(X_whole, y_whole, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X_whole, y_whole, test_size=test_size, random_state = random_state)
    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)
    
    params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}
    generations = 1000
    evals = [(dtest, "validation"), (dtrain, "train")]
    model = xgb.train(params = params, dtrain = dtrain, num_boost_round = generations, evals = evals, verbose_eval = 50, early_stopping_rounds = 50)

    predictions = model.predict(dtest)
    rmse = mean_squared_error(y_test, predictions, squared=False)

    print(f"RMSE of the XGBoost: {rmse:.3f}")

XGBoost(stats, result, 0.25, 16)
# RMSE score of 3.759 on testing data

'''
Clustering:

Team Similarity: Group teams with similar performance metrics using clustering algorithms like K-means or hierarchical clustering. This can help identify teams with similar styles or strengths.
'''

'''
Feature Importance and Analysis:

Feature Analysis: Determine which features most influence game outcomes or team performance using feature importance techniques from models like random forests or gradient boosting.
'''

'''
Time Series Analysis:

Performance Trends: Analyze how team performance metrics change over time. You could use time series analysis methods or recurrent neural networks (RNNs) to capture temporal patterns and trends.
'''
'''
Simulation and Scenario Analysis:

Game Simulations: Use the data to simulate various game scenarios and outcomes to understand potential strategies or outcomes.
'''

# Getting the statistics for every team
def get_team_stat(years):
    team_data = {}
    # EX: 'GSW' : [winCt, LossCt, teamPTS, teamAST, teamTO, teamSTL, teamBLK, teamPF, teamFGA, teamFGM, teamFG%, 
    # team2PA, team2PM, team2P%, team3PA, team3PM, team3P%, teamFTA, teamFTM, teamFT%, teamORB, teamDRB, teamTRB, teamOrtg, teamDrtg, teamGames]
    for index, row in df.iterrows():
        if (row["year"] in years):
            if (row["teamAbbr"] not in team_data):
                team_data[row["teamAbbr"]] = [0 for i in range(0, 26)]
            
            if (row["teamRslt"] == "Win"):
                    team_data[row["teamAbbr"]][0] += 1
            else:
                    team_data[row["teamAbbr"]][1] += 1
            team_data[row["teamAbbr"]][2] += row["teamPTS"]
            team_data[row["teamAbbr"]][3] += row["teamAST"]
            team_data[row["teamAbbr"]][4] += row["teamTO"]
            team_data[row["teamAbbr"]][5] += row["teamSTL"]
            team_data[row["teamAbbr"]][6] += row["teamBLK"]
            team_data[row["teamAbbr"]][7] += row["teamPF"]
            team_data[row["teamAbbr"]][8] += row["teamFGA"]
            team_data[row["teamAbbr"]][9] += row["teamFGM"]
            team_data[row["teamAbbr"]][10] += row["teamFG%"]
            team_data[row["teamAbbr"]][11] += row["team2PA"]
            team_data[row["teamAbbr"]][12] += row["team2PM"]
            team_data[row["teamAbbr"]][13] += row["team2P%"]
            team_data[row["teamAbbr"]][14] += row["team3PA"]
            team_data[row["teamAbbr"]][15] += row["team3PM"]
            team_data[row["teamAbbr"]][16] += row["team3P%"]
            team_data[row["teamAbbr"]][17] += row["teamFTA"]
            team_data[row["teamAbbr"]][18] += row["teamFTM"]
            team_data[row["teamAbbr"]][19] += row["teamFT%"]
            team_data[row["teamAbbr"]][20] += row["teamORB"]
            team_data[row["teamAbbr"]][21] += row["teamDRB"]
            team_data[row["teamAbbr"]][22] += row["teamTRB"]
            team_data[row["teamAbbr"]][23] += row["teamOrtg"]
            team_data[row["teamAbbr"]][24] += row["teamDrtg"]
            team_data[row["teamAbbr"]][25] += 1

                
    print(team_data)
    return team_data
#get_team_stat([2013])