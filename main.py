# A project dedicated to my glorious king the great Wardell Steph Curry
# Comparing Steph Curry to other players from 2012-18
import pandas as pd
import matplotlib.pyplot as plt
import requests 
from bs4 import BeautifulSoup
import regex as re
whole_df = pd.read_csv("archive\\2012-18_playerBoxScore.csv")
#print(whole_df.describe())
years = [2012, 2013, 2014, 2015, 2016, 2017, 2018]
# Gets a player's stats in a given year.
def getPlayerStats(df, firstName, lastName, teamAbbr, year):
    df['gmDate'] = pd.to_datetime(df['gmDate'])
    df['year'] = df['gmDate'].dt.year
    player_df = df.loc[(df['playLNm'] == lastName) & (df["playFNm"] == firstName) & (df["year"].isin(year))]
    #print(player_df)
    return player_df

curry = getPlayerStats(whole_df, "Wardell", "Curry", "GS", years)

# Getting the stats for all players of a certain position in a certain year.
def getPositionStats(df, position, year):
    df['gmDate'] = pd.to_datetime(df['gmDate'])
    df['year'] = df['gmDate'].dt.year
    position_df = df.loc[(df['playPos'] == position) & (df["year"].isin(year))]
    #print(position_df)
    return position_df

# Getting all players within a certain weight and height class of a certain year. 
def getWeightHeightStats(df, weight, height, dw, dh, year):
    df['gmDate'] = pd.to_datetime(df['gmDate'])
    df['year'] = df['gmDate'].dt.year
    wh_df = df.loc[(df['playWeight'] >= weight - dw) & (df['playWeight'] <= weight + dw) & 
                   (df['playHeight'] >= height - dh) & (df['playHeight'] <= height + dh) & 
                   (df['year'].isin(year))]
    #print(wh_df)
    return wh_df

# Getting an attribute for all players in a dataframe per minute
def getAttrPerMin(df, attr):
    df['fullName'] = df['playFNm'] + ' ' + df['playLNm']
    df['attrPerMin'] = df[attr] / df['playMin']
    grouped_df = df.groupby('fullName', as_index=False)['attrPerMin'].mean()
    #print(attribute)
    return grouped_df

# Getting an attribute for all players in a dataframe per game
def getAttrPerGame(df, attr):
    df['fullName'] = df['playFNm'] + ' ' + df['playLNm']
    df['attrPerGame'] = df[attr]
    grouped_df = df.groupby('fullName', as_index=False)["attrPerGame"].mean()
    return grouped_df

# Displays a certain attribute from 2 dataframes in a barchart
def displayPerMinute(df1, df2, attribute, title, unit):
    plt.figure(figsize=(12, 6))

    if unit == "min":
        curry_df = getAttrPerMin(df1, attribute).sort_values(by='attrPerMin').dropna()
        curry_df = curry_df[curry_df['attrPerMin'] != 0]
        
        pg_df = getAttrPerMin(df2, attribute).sort_values(by='attrPerMin').dropna()
        pg_df = pg_df[pg_df['attrPerMin'] != 0]
    
        combined_df = pd.concat([pg_df, curry_df]).sort_values(by='attrPerMin')

        plt.bar(combined_df['fullName'], combined_df['attrPerMin'], color='skyblue', label='PG Data')
        plt.bar(curry_df['fullName'], curry_df['attrPerMin'], color='black', label='Curry Data')
    else:
        curry_df = getAttrPerGame(df1, attribute).sort_values(by='attrPerGame').dropna()
        curry_df = curry_df[curry_df['attrPerGame'] != 0]
        
        pg_df = getAttrPerGame(df2, attribute).sort_values(by='attrPerGame').dropna()
        pg_df = pg_df[pg_df['attrPerGame'] != 0]
        
        combined_df = pd.concat([pg_df, curry_df]).sort_values(by='attrPerGame')
        plt.bar(combined_df['fullName'], combined_df['attrPerGame'], color='skyblue', label='PG Data')
        plt.bar(curry_df['fullName'], curry_df['attrPerGame'], color='black', label='Curry Data')

    
    plt.xlabel('Player Name')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(rotation=90)
    #plt.legend()
    plt.tight_layout()
    plt.show()

# Displaying charts for similar point guards
def pgCompare(years):
    pg = getPositionStats(whole_df,"PG", years)
    # Getting the points per game in the 2015 season--------------------------------
    displayPerMinute(curry, pg, "playPTS", "Curry vs Point Guards, 2015, Points Per Min", "min")
    # Getting the assists per game in the 2015 season--------------------------------
    displayPerMinute(curry, pg, "playAST", "Curry vs Point Guards, 2015, Assists Per Min", "min")
    # Getting the turnovers per game in the 2015 season--------------------------------
    displayPerMinute(curry, pg, "playTO", "Curry vs Point Guards, 2015, Turnovers Per Min", "min")
    # Getting the steals per game in the 2015 season--------------------------------
    displayPerMinute(curry, pg, "playSTL", "Curry vs Point Guards, 2015, Steals Per Min", "min")
    # Getting the blocks per game in the 2015 season--------------------------------
    displayPerMinute(curry, pg, "playBLK", "Curry vs Point Guards, 2015, Blocks Per Min", "min")

    # Getting the personal fouls per game in the 2015 season--------------------------------
    displayPerMinute(curry, pg, "playPF", "Curry vs Point Guards, 2015, Personal Fouls Per Min", "min")

    # Getting the FGA per game in the 2015 season--------------------------------
    displayPerMinute(curry, pg, "playFGA", "Curry vs Point Guards, 2015, FGA Per Min", "min")

    displayPerMinute(curry, pg, "playFGM", "Curry vs Point Guards, 2015, FGM Per Min", "min")
    displayPerMinute(curry, pg, "playFG%", "Curry vs Point Guards, 2015, FG% Per Game", "game")

    displayPerMinute(curry, pg, "play2PA", "Curry vs Point Guards, 2015, 2PA Per Min", "min")
    displayPerMinute(curry, pg, "play2PM", "Curry vs Point Guards, 2015, 2PM Per Min", "min")
    displayPerMinute(curry, pg, "play2P%", "Curry vs Point Guards, 2015, 2P% Per Game", "game")

    displayPerMinute(curry, pg, "play3PA", "Curry vs Point Guards, 2015, 3PA Per Min", "min")
    displayPerMinute(curry, pg, "play3PM", "Curry vs Point Guards, 2015, 3PM Per Min", "min")
    displayPerMinute(curry, pg, "play3P%", "Curry vs Point Guards, 2015, 3P% Per Game", "game")

    displayPerMinute(curry, pg, "playFTA", "Curry vs Point Guards, 2015, FTA Per Min", "min")
    displayPerMinute(curry, pg, "playFTM", "Curry vs Point Guards, 2015, FTM Per Min", "min")
    displayPerMinute(curry, pg, "playFT%", "Curry vs Point Guards, 2015, FT% Per Game", "game")

    displayPerMinute(curry, pg, "playORB", "Curry vs Point Guards, 2015, ORB Per Min", "min")
    displayPerMinute(curry, pg, "playDRB", "Curry vs Point Guards, 2015, DRB Per Min", "min")
    displayPerMinute(curry, pg, "playTRB", "Curry vs Point Guards, 2015, TRB Per Min", "min")

# Displaying charts for similar physical builds
def whCompare(years):
    sd_height = whole_df["playHeight"].std()
    sd_weight = whole_df["playWeight"].std()
    #print(sd_height, sd_weight)
    # Getting the players who are within 1 standard deviation of Curry's height and weight. 
    wh_df = getWeightHeightStats(whole_df,185, 75, sd_weight, sd_height, years)
    # Getting the points per game in the 2015 season--------------------------------
    displayPerMinute(curry, wh_df, "playPTS", "Curry vs Similar Builds, 2015, Points Per Min", "min")
    # Getting the assists per game in the 2015 season--------------------------------
    displayPerMinute(curry, wh_df, "playAST", "Curry vs Similar Builds, 2015, Assists Per Min", "min")
    # Getting the turnovers per game in the 2015 season--------------------------------
    displayPerMinute(curry, wh_df, "playTO", "Curry vs Similar Builds, 2015, Turnovers Per Min", "min")
    # Getting the steals per game in the 2015 season--------------------------------
    displayPerMinute(curry, wh_df, "playSTL", "Curry vs Similar Builds, 2015, Steals Per Min", "min")
    # Getting the blocks per game in the 2015 season--------------------------------
    displayPerMinute(curry, wh_df, "playBLK", "Curry vs Similar Builds, 2015, Blocks Per Min", "min")

    # Getting the personal fouls per game in the 2015 season--------------------------------
    displayPerMinute(curry, wh_df, "playPF", "Curry vs Similar Builds, 2015, Personal Fouls Per Min", "min")

    # Getting the FGA per game in the 2015 season--------------------------------
    displayPerMinute(curry, wh_df, "playFGA", "Curry vs Similar Builds, 2015, FGA Per Min", "min")

    displayPerMinute(curry, wh_df, "playFGM", "Curry vs Similar Builds, 2015, FGM Per Min", "min")
    displayPerMinute(curry, wh_df, "playFG%", "Curry vs Similar Builds, 2015, FG% Per Game", "game")

    displayPerMinute(curry, wh_df, "play2PA", "Curry vs Similar Builds, 2015, 2PA Per Min", "min")
    displayPerMinute(curry, wh_df, "play2PM", "Curry vs Similar Builds, 2015, 2PM Per Min", "min")
    displayPerMinute(curry, wh_df, "play2P%", "Curry vs Similar Builds, 2015, 2P% Per Game", "game")

    displayPerMinute(curry, wh_df, "play3PA", "Curry vs Similar Builds, 2015, 3PA Per Min", "min")
    displayPerMinute(curry, wh_df, "play3PM", "Curry vs Similar Builds, 2015, 3PM Per Min", "min")
    displayPerMinute(curry, wh_df, "play3P%", "Curry vs Similar Builds, 2015, 3P% Per Game", "game")

    displayPerMinute(curry, wh_df, "playFTA", "Curry vs Similar Builds, 2015, FTA Per Min", "min")
    displayPerMinute(curry, wh_df, "playFTM", "Curry vs Similar Builds, 2015, FTM Per Min", "min")
    displayPerMinute(curry, wh_df, "playFT%", "Curry vs Similar Builds, 2015, FT% Per Game", "game")

    displayPerMinute(curry, wh_df, "playORB", "Curry vs Similar Builds, 2015, ORB Per Min", "min")
    displayPerMinute(curry, wh_df, "playDRB", "Curry vs Similar Builds, 2015, DRB Per Min", "min")
    displayPerMinute(curry, wh_df, "playTRB", "Curry vs Similar Builds, 2015, TRB Per Min", "min")

# Fetches all all-star data for a certain year
def getAllStars(year):
    url = f'https://www.basketball-reference.com/leagues/NBA_{year}.html#all_all_star_game_rosters'
    
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find the All-Star Game Rosters section
        all_star_rosters = soup.find(id='all_all_star_game_rosters')
        raw = str(all_star_rosters)
        raw = str(re.sub('<[^<]+?>', '', str(all_star_rosters)))
        raw = raw.replace("&nbsp;", ", ").replace("*", "").replace(", -->", "").replace("All-Star Game Rosters", "").replace(", ,", ", ")
        raw = " ".join(raw.split())
        #print(all_star_rosters.prettify())\
        #print(raw)
    else:
        print('Failed to retrieve data', response.status_code)
    list_of_players = raw.split(", ")
    #print(list_of_players)
    allStars = []
    for ele in list_of_players:
        if all(c.isalpha() or c.isspace() for c in ele.replace(" ", "")) and ele != "West " and len(ele)>=2:
            #print(ele)
            allStars.append(ele)
    return allStars

# Displaying charts for all all stars
def compareAllStars(df2):
    # Getting the points per game in the 2015 season--------------------------------
    displayPerMinute(curry, df2, "playPTS", "Curry vs All Stars, 2015, Points Per Min", "min")
    # Getting the assists per game in the 2015 season--------------------------------
    displayPerMinute(curry, df2, "playAST", "Curry vs All Stars, 2015, Assists Per Min", "min")
    # Getting the turnovers per game in the 2015 season--------------------------------
    displayPerMinute(curry, df2, "playTO", "Curry vs All Stars, 2015, Turnovers Per Min", "min")
    # Getting the steals per game in the 2015 season--------------------------------
    displayPerMinute(curry, df2, "playSTL", "Curry vs All Stars, 2015, Steals Per Min", "min")
    # Getting the blocks per game in the 2015 season--------------------------------
    displayPerMinute(curry, df2, "playBLK", "Curry vs All Stars, 2015, Blocks Per Min", "min")

    # Getting the personal fouls per game in the 2015 season--------------------------------
    displayPerMinute(curry, df2, "playPF", "Curry vs All Stars, 2015, Personal Fouls Per Min", "min")

    # Getting the FGA per game in the 2015 season--------------------------------
    displayPerMinute(curry, df2, "playFGA", "Curry vs All Stars, 2015, FGA Per Min", "min")

    displayPerMinute(curry, df2, "playFGM", "Curry vs All Stars, 2015, FGM Per Min", "min")
    displayPerMinute(curry, df2, "playFG%", "Curry vs All Stars, 2015, FG% Per Game", "game")

    displayPerMinute(curry, df2, "play2PA", "Curry vs All Stars, 2015, 2PA Per Min", "min")
    displayPerMinute(curry, df2, "play2PM", "Curry vs All Stars, 2015, 2PM Per Min", "min")
    displayPerMinute(curry, df2, "play2P%", "Curry vs All Stars, 2015, 2P% Per Game", "game")

    displayPerMinute(curry, df2, "play3PA", "Curry vs All Stars, 2015, 3PA Per Min", "min")
    displayPerMinute(curry, df2, "play3PM", "Curry vs All Stars, 2015, 3PM Per Min", "min")
    displayPerMinute(curry, df2, "play3P%", "Curry vs All Stars, 2015, 3P% Per Game", "game")

    displayPerMinute(curry, df2, "playFTA", "Curry vs All Stars, 2015, FTA Per Min", "min")
    displayPerMinute(curry, df2, "playFTM", "Curry vs All Stars, 2015, FTM Per Min", "min")
    displayPerMinute(curry, df2, "playFT%", "Curry vs All Stars, 2015, FT% Per Game", "game")

    displayPerMinute(curry, df2, "playORB", "Curry vs All Stars, 2015, ORB Per Min", "min")
    displayPerMinute(curry, df2, "playDRB", "Curry vs All Stars, 2015, DRB Per Min", "min")
    displayPerMinute(curry, df2, "playTRB", "Curry vs All Stars, 2015, TRB Per Min", "min")

# Gets all the all-stars between a range of years
def getStars(years):
    list_of_stars = []
    for year in years:
        list_of_stars += getAllStars(year)

    df_list = []
    print(list_of_stars)

    for star in list_of_stars:
        first_name, last_name = star.split()[0], star.split()[1]
        print("ALL STAR NAME: ", first_name, last_name)
        
        player_stats = getPlayerStats(whole_df, first_name, last_name, "", years)
        player_stats_df = player_stats if isinstance(player_stats, pd.DataFrame) else pd.DataFrame()
        
        df_list.append(player_stats_df)

    # Concatenate all DataFrames in the list into a single DataFrame
    all_star_df = pd.concat(df_list, ignore_index=True)

    print(all_star_df)
    return all_star_df


### COMPARING TO SIMILAR WEIGHT/HEIGHT ###
whCompare([2015])

### COMPARING TO OTHER PGs ###
pgCompare([2015])

### COMPARING TO OTHER ALL STARS OF THE YEAR
all_stars = getStars([2017, 2018])
compareAllStars(all_stars)