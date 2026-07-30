install.packages("fitzRoy")
library(dplyr)
library(fitzRoy)

fetch_player_stats(2024, comp = "AFL")

fetch_player_stats_afl(season = 2025, round_number = NULL, comp = "AFLM")


fetch_supercoach_scores(year = 2025, rounds = 1:30)

#install.packages("devtools")
#devtools::install_github("jimmyday12/fitzRoy")
#library(fitzRoy)
#library(dplyr)
#library(ggplot2)

#stats <- get_afltables_stats(start_date = "2000-01-01", end_date = "2025-02-1")
#footywire_data <- update_footywire_stats()
#match_results <- get_match_results()


#currentDate <- format(Sys.time(), "%Y-%m-%d_%H-%M")
#csvFileName <- paste("match_results ",currentDate,".csv",sep="") 
#write.csv(match_results, csvFileName)
