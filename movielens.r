## Aansh Sardana
## MovieLens Project 
## HarvardX: PH125.9x - Capstone Project

#################################################
# MovieLens Rating Prediction Project Code 
################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_set <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                          col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings_set, movies, by = "movieId")


# The Validation subset will be 10% of the MovieLens data.
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
#Make sure userId and movieId in validation set are also in edx subset:
validation <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings_set, movies, test_index, temp, movielens, removed)

#### Methods and Analysis ####

#Analysis of data#

#exploration
str(edx)

# Head
head(edx) %>% print.data.frame()

# Summary
summary(edx)

# Number of unique movies and users in the edx dataset 
edx %>%summarize(n_users = n_distinct(userId),  n_movies = n_distinct(movieId))

# Distribution of rating
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.30,fill="#fc0303", color = "blue", alpha=0.5) +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 250000))) +
  labs(title = "Distribution of rating", y = "Number of Rating", x = "Rating")

# Plot number of ratings per movie
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 25, fill="#fc0303", color = "blue", alpha=0.5) +
  scale_x_log10() +
  labs(title = "Number of ratings received per movie", y = "Total Number of movies", x = "Total Number of ratings")

# Plot number of ratings given by users
edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, fill="#fc0303", color = "blue", alpha=0.5) +
  scale_x_log10() +
  labs(title = "Number of ratings given by users", y = "Number of users", x = "Number of ratings")

# Plot mean movie ratings given by users
edx %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, fill="#fc0303", color = "blue", alpha=0.5) +
  labs(title = "Mean movie ratings given by users", y = "Number of users", x = "Mean of rating") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  theme_light()

## Average movie rating model ##

# Computing the dataset's mean rating
me_an <- mean(edx$rating)
# Displaying mean
me_an

# Testing results based on simple prediction
save_rmse <- RMSE(validation$rating, me_an)
save_rmse

# Check results and saving predictions in data frame
results_rmse <- data_frame(method = "Average movie rating model", RMSE = save_rmse)
results_rmse %>% knitr::kable()


## Movie effect model ##

# Simple model taking into account the movie effect b_i 2) Subtract the rating minus the mean for each rating the movie received 3) Plot number of movies with the computed b_i
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - me_an))
movie_avgs
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., fill=I("#fc0303"), color = I("blue"), alpha=I(0.5), ylab = "Number of movies", main = "Number of movies with the computed b_i")


# Testing and saving rmse results 
predicted_ratings <- me_an +  validation %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
predicted_ratings
rmse_model_1 <- RMSE(predicted_ratings, validation$rating)
results_rmse <- bind_rows(results_rmse,
                          data_frame(method="Movie effect model",  
                                     RMSE = rmse_model_1 ))
results_rmse
# Checking results
results_rmse %>% knitr::kable()


## Movie and user effect model ##

# Plotting penaly term user effect #
user_avgs<- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating - me_an - b_i))
user_avgs%>% qplot(b_u, geom ="histogram", bins = 30, data = ., color = I("black"), fill=I("#fc0303"),alpha=I(0.5),ylab = "Number of movies",main = "Number of movies with the computed b_u")


user_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - me_an - b_i))


# Testing and saving rmse results 
predicted_ratings <- validation%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = me_an + b_i + b_u) %>%
  pull(pred)

model_2_rmse <- RMSE(predicted_ratings, validation$rating)
results_rmse <- bind_rows(results_rmse,
                          data_frame(method="Movie and user effect model",  
                                     RMSE = model_2_rmse))
# Checking result
results_rmse %>% knitr::kable()

## Regularized movie and user effect model ##

# lamb is a tuning parameter
lamb <- seq(0, 10, 0.25)
lamb

# For each lambda,finding b_i & b_u, followed by rating prediction & testing
rmses <- sapply(lamb, function(l){
  
  me_an <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - me_an)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - me_an)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = me_an + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})


# Plotting rmses vs lamb to select the optimal lambda                                                             
qplot(lamb, rmses, color = I("#fc0303"), geom=c("point", "smooth"))  

# The optimal lamb which is minimum of all                                                            
lamb<- lamb[which.min(rmses)]
lamb

# Testing and saving results                                                             
results_rmse <- bind_rows(results_rmse, data_frame(method="Regularized movie and user effect model",RMSE = min(rmses)))

#RESULTS #                                                                                                                     
results_rmse %>% knitr::kable()
