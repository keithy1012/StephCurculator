import pandas as pd

# Aggregates a player's stats by month and year. 
whole_df = pd.read_csv("archive\\2012-18_playerBoxScore.csv")
whole_df["year"] = whole_df["gmDate"].str.slice(0, 4)
whole_df["month"] = whole_df["gmDate"].str.slice(5, 7)
# Returns a new df with year | player name | total pts | total assists ...
def agg_stats (df):
    aggregated_df = pd.DataFrame()
    aggregated_df = df.groupby(["year", "month", "playDispNm"]).agg({
        "playPTS" : "sum",
        "playAST" : "sum",
        "playTO"  : "sum",
        "playSTL" : "sum",
        "playBLK" : "sum",
        "playFGM"  : "sum",     
        "play2PA" : "sum",
        "play2PM" : "sum",
        "play3PA"  : "sum",   
        "play3PM" : "sum",
        "playFTA"  : "sum",          
        "playFTM"  : "sum",    
    }).reset_index()
    aggregated_df.columns = ["year", "month", "player_name", "playPTS", "playAST", 
                             "playTO", "playSTL", "playBLK", "playFGM", "play2PA", 
                             "play2PM", "play3PA", "play3PM", "playFTA", "playFTM"]
    print(aggregated_df)
    return aggregated_df

aggregated_data = agg_stats(whole_df)

aggregated_data.to_csv("aggregated_data_by_year.csv")
