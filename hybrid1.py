import pandas as pd
from nltk.tokenize import RegexpTokenizer
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from  scipy.sparse.linalg import svds
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# from sklearn.model_selection import train_test_split

def eval_class(alist):
	count = 0
	for i,j in enumerate(alist):
		if j <= 3.5:
			alist[i] = 0
		elif j > 3.5:
			alist[i] = 1
			count +=1
	return alist, count



def evaluate(array_data1,predicted_rating, userId, countd):
	predicted_rating1 = predicted_rating
	# print(predicted_rating)
	pp_11 = predicted_rating.iloc[userId-1].sort_values(ascending=False).head(10)
	# print(pp_11)
	array_1 = array_data1.iloc[userId-1].sort_values(ascending=False)

	movie_id = []
	pred_value = []
	pp_1 = pp_11.reset_index()

	# print(pp_1[userId].values)

	for i in range(len(pp_1)):
		movie_id.append(pp_1['movieId'][i])
	# print(movie_id)
	true_value = []
	# print()

	for i in movie_id:
		true_value.append(array_1[i])
	# print(pred_value)
	# print(true_value)

	for i in movie_id:
		pred_value.append(pp_11[i])
	# print(pred_value)
	value, count1= eval_class(pred_value)
	true_value, count= eval_class(true_value)
	print(true_value)
	print(count1)
	precision = count / count1
	print(countd)
	# print(value)
	#
	# precision = precision_score(value, pred_value)
	print(precision)

	recall = count/countd
	print(recall)


def colab_recom(userId):



	def C_recommender(predicted_rating, userId, array):
		# print(predicted_rating.iloc[4].sort_values(ascending = False))

		predicted_rating = predicted_rating.iloc[(userId-1)]


		new_data = data_rating[data_rating.userId == (userId)]

		new_full = new_data.merge(data_movies,  on= 'movieId')
		new_full = new_full.sort_values(['rating'], ascending=False)
		print(new_full)
		seen_recommendations = data_movies
		seen_recommendations = seen_recommendations.merge(pd.DataFrame(predicted_rating).reset_index(), on = 'movieId')
		# print(seen_recommendations.sort_values(userId,ascending=False).head(10))
		# print(new_full)

		recommendations = (data_movies[~data_movies['movieId'].isin(new_full['movieId'])])
		recommendations = recommendations.merge(pd.DataFrame(predicted_rating).reset_index(), on = 'movieId')
		# print(recommendations)

		recommendations = recommendations.sort_values(userId, ascending=False)
		return recommendations, seen_recommendations, array
		# for i in range(num_rccom):
			# print(recommendations)
		print

	data_rating = pd.read_csv("ratings.csv")
	data_rating = data_rating.drop('timestamp', axis=1)

	meta_data = pd.read_csv('movies_metadata.csv', low_memory = False)


	group_user_ratings = data_rating.groupby('userId')['rating']


	data_rating.dropna(inplace=True)


	pivot_data = data_rating.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)

	array_data = pivot_data.values
	data_rating_mean = np.mean(array_data, axis = 1)
	array_normalized = array_data - data_rating_mean.reshape(-1,1)

	U, sigma, Vt = svds(array_normalized, k = 30)

	sigma = np.diag(sigma)# make the diagonal matrix form of sigma

	predicted_rating = np.dot(np.dot(U, sigma), Vt) + data_rating_mean.reshape(-1,1)

	predicted_rating = pd.DataFrame(predicted_rating, columns = pivot_data.columns, index = pivot_data.index)
	array_data1 = pd.DataFrame(array_data, columns = pivot_data.columns, index = pivot_data.index)

	data_movies = pd.read_csv("movies1.csv")
	data_movies['vote_average'] = meta_data['vote_average']
	data_movies['vote_count'] = meta_data['vote_count']

	array_1 = array_data1.iloc[userId-1].sort_values(ascending=False)
	# print(array_1[608])
	true_rel, countd = eval_class(array_1)
	# print(countd)
	evaluate(array_data1, predicted_rating, userId, countd)



	data_movies = pd.DataFrame(data_movies, columns = ['movieId','title','genres','vote_average','vote_count'])
	# print(data_movies)
	return C_recommender(predicted_rating, userId, array_data1)

def main_hybrid(clb, userId):
	def weight(hb):
		v = hb['vote_count']
		r = hb['vote_average']
		ur = hb[userId]

		return (ur*2.8 + (((v/(v+quant) *r) + (quant/(quant+v)) * avg)))/2
	vote_counts = clb[clb['vote_count'].notnull()]['vote_count'].astype('int')
	vote_avg = clb[clb['vote_average'].notnull()]['vote_average'].astype('int')
	user_rate = clb[clb[userId].notnull()][userId].astype('float')
	# print(user_rate)
	avg = vote_avg.mean()
	mean_bar = user_rate.mean()

	quant = vote_counts.quantile(0.6)
	hybrid = clb[(clb['vote_count'] >= quant) & \
	(clb['vote_count'].notnull()) & (clb['vote_average'].notnull())]
	# hybrid = clb[(clb['vote_count'].notnull()) & (clb['vote_average'].notnull())]
	# print(hybrid)
	hybrid = hybrid.astype({"vote_count":"int",userId:"float","vote_average":"int"})

	hybrid['weight'] = hybrid.apply(weight, axis =1)
	hybrid_predicted_unseen = hybrid.sort_values('weight',ascending = False)
	# for i in range(83):
	print(hybrid_predicted_unseen.head(10))
	return hybrid_predicted_unseen

def hybrid(CollabW, userId):
	def evaluate1(array_data1,predicted_rating, userId, countd):
		predicted_rating1 = predicted_rating
		pp_11 = pd.DataFrame(predicted_rating1, columns = ['movieId', userId ]).head(10)
		# pp_11 = predicted_rating.iloc[userId-1].sort_values(ascending=False).head(10)
		array_1 = array_data1.iloc[userId-1].sort_values(ascending=False)
		# print(array_1)
		# array_2 = array_data1.iloc[userId].sort_values()
		# print(array_2.head(20))
		movie_id = []
		true_value = []
		# pp_1 = pp_11.reset_index()
		# print(pp_11['movieId'].values)
		for i in pp_11['movieId'].values:
			movie_id.append(i)
#
		# print(pp_11[userId])

		pred_value = []
		# # print()
		#
		for i in movie_id:
			true_value.append(array_1[i])
		# print(pred_value)

		for i in pp_11[userId].values:
			pred_value.append(i)
		# print(value)
		print(pred_value)
		pred_value, count1= eval_class(pred_value)
		true_value, count= eval_class(true_value)
		print(true_value)
		print(count1)
		# precision = count / count1
		print(countd)
		# print(value)
		#
		# precision = precision_score(value, pred_value)
		# print(precision)

		recall = count/countd
		print(recall)




	Collab_prec, seen_colab, array= colab_recom(userId)
	print(seen_colab.sort_values(userId,ascending=False).head(10))
	# Collab_prec = Collab_prec.head(200)
	# print(type(c))
	# Collab_prec.rename(columns = {userId})
	array_1 = array.iloc[userId-1].sort_values(ascending=False)
	# print(array_1[608])
	true_rel, countd = eval_class(array_1)
	pred = main_hybrid(Collab_prec.sort_values(userId,ascending=False).head(200), userId)
	# score[]
	evaluate1(array, pred, userId, countd)



	# print(colab)
	# print(data_movies)





# Crecomm()
# colab_recom()
hybrid(200, 46)
