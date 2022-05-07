import pandas as pd
import numpy as np
from  scipy.sparse.linalg import svds
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.reader import Reader

from sklearn.metrics import mean_squared_error

def C_recommender(predicted_rating, userId, num_rccom):
	predicted_rating = predicted_rating.iloc[(userId-1)]
	# print(predicted_rating)


	new_data = data_rating[data_rating.userId == (userId)].head(10)

	new_full = new_data.merge(data_movies,  on= 'movieId')
	new_full = new_full.sort_values(['rating'], ascending=False)
	print(new_full)
	seen_recommendations = data_movies
	seen_recommendations = seen_recommendations.merge(pd.DataFrame(predicted_rating).reset_index(), on = 'movieId')
	print(new_full.sort_values('rating', ascending= False))

	recommendations = (data_movies[~data_movies['movieId'].isin(new_full['movieId'])])
	recommendations = recommendations.merge(pd.DataFrame(predicted_rating).reset_index(), on = 'movieId')
	# print(recommendations)

	recommendations = recommendations.sort_values(userId, ascending=False)

	# for i in range(num_rccom):
	print("Recommendations for user ", userId)
	# print(seen_recommendations.head(10))
	print(recommendations.head(10))
	print


data_rating = pd.read_csv("ratings.csv")
data_rating = data_rating.drop('timestamp', axis=1)



group_user_ratings = data_rating.groupby('userId')['rating']


data_rating.dropna(inplace=True)


pivot_data = data_rating.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)

array_data = pivot_data.values
# print(array_data)
data_rating_mean = np.mean(array_data, axis = 1)
array_normalized = array_data - data_rating_mean.reshape(-1,1)

U, sigma, Vt = svds(array_normalized, k = 30)

sigma = np.diag(sigma)# make the diagonal matrix form of sigma

predicted_rating = np.dot(np.dot(U, sigma), Vt) + data_rating_mean.reshape(-1,1)
print(predicted_rating)

predicted_rating = pd.DataFrame(predicted_rating, columns = pivot_data.columns, index = pivot_data.index)

data_movies = pd.read_csv("movies1.csv")
# print(array_data)
meta_data = pd.read_csv("movies_metadata.csv")
data_movies['vote_average'] = meta_data['vote_average']
data_movies['vote_count'] = meta_data['vote_count']

# data_movies = pd.DataFrame(data_movies, columns = ['movieId','title','genres','vote_average','vote_count'])

data_movies = pd.DataFrame(data_movies, columns = ['movieId','title','genres'])
# print(data_movies)
C_recommender(predicted_rating, 46, 10)
reader = Reader()
data = Dataset.load_from_df(data_rating[['userId', 'movieId', 'rating']], reader)
# data = data.split(n_folds=5)
svd = SVD()
# evaluate(svd, data, measures=['RMSE', 'MAE'])
cross_validate(svd, data, measures=['RMSE', 'MSE'], cv = 5, verbose =True)
#
# rms = mean_squared_error(array_data, predicted_rating, squared=False)
# mse = mean_squared_error(array_data, predicted_rating)
# print("RMSE", rms)
# print("MSE", mse)
