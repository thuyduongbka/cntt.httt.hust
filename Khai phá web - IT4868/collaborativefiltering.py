import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy import sparse 

import joblib

class CF(object):

    def __init__(self, Y_data, k, dist_func = cosine_similarity, uuCF = 1):
        self.uuCF = uuCF # user-user (1) or item-item (0) CF
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        self.k = k
        self.dist_func = dist_func
        self.Ybar_data = None
        # số lượng user/item
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1 
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1
    
    def add(self, new_data):
        self.Y_data = np.concatenate((self.Y_data, new_data), axis = 0)
    
    def normalize_Y(self):
        users = self.Y_data[:, 0] 
        #lấy ra tất cả các user
        self.Ybar_data = self.Y_data.copy() 
        #đây sẽ là mảng để cập nhật các ratting thành các giá trị chuẩn hóa
        self.mu = np.zeros((self.n_users,)) 
        #mảng lưu các giá trị đánh giá trung bình của user
        for n in range(self.n_users):
            # Lấy chỉ số (index) của user n
            ids = np.where(users == n)[0].astype(np.int32)
            # Lấy id của item được user n đánh giá
            item_ids = self.Y_data[ids, 1] 
            # Lấy rating của các item được user n đánh giá
            ratings = self.Y_data[ids, 2]
            # tính trung bình các giá trị rating của người dùng u Ru
            m = np.mean(ratings) 
            if np.isnan(m):
                m = 0 
            self.mu[n] = m
            # cập nhật giá chị chuẩn hóa  = ratings - giá trị trung bình
            self.Ybar_data[ids, 2] = ratings - self.mu[n]

        # Đưa về dạng (item,user) = value
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
            (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users)).tocsr()
     

    def similarity(self):     
         # Ybar.T ma trận chuyển vị
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)
    
        
    def refresh(self):        
        self.normalize_Y()
        self.similarity() 
        
    def fit(self):
        self.refresh()
        
    # dự đoán ratting của user u cho item i
    def __pred(self, u, i, normalized = 1):
        
        # index users đã rated i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)

        # id users who đã i
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)

        # similarity u vs user đã rated i
        sim = self.S[u, users_rated_i]
       
        # Step 4: lấy ra id của k user giống nhất
        a = np.argsort(sim)[-self.k:]        

        # lấy giá trị similar của k user giống nhất
        nearest_s = sim[a]
     
        # lấy giá trị rating item i của k user giống nhất
        r = self.Ybar[i, users_rated_i[a]]
        
        if normalized:
            # add a small number, for instance, 1e-8, to avoid dividing by 0
            return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8)

        return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8) + self.mu[u]
    
    def pred(self, u, i, normalized = 1):        
        if self.uuCF: return self.__pred(u, i, normalized)
        return self.__pred(i, u, normalized)
            
    
    def recommend(self, u):
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()              
        recommended_items = []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                if rating > 0: 
                    recommended_items.append(i)
        
        return recommended_items 
    
    def print_recommendation(self):
        print ('Recommendation: ')
        for u in range(self.n_users):
            recommended_items = self.recommend(u)
            if self.uuCF:
                print ('    Recommend item(s):', recommended_items, 'for user', u)
            else: 
                print ('    Recommend item', u, 'for user(s) : ', recommended_items)



r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

# ratings_base = pd.read_csv('ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
# ratings_test = pd.read_csv('ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')

# rate_train = ratings_base.to_numpy()
# rate_test = ratings_test.to_numpy()

data = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')
rate_train, rate_test = train_test_split(data.to_numpy(),test_size=0.25)

# indices start from 0
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1

print ('Number of traing rates:', rate_train.shape[0])
print ('Number of test rates:', rate_test.shape[0])

#--------------------------------------------
# user based

rs = CF(rate_train, k = 30, uuCF = 1)
rs.fit()

joblib.dump(rs, 'uucf')

n_tests = rate_test.shape[0]
SE = 0 # squared error
for n in range(n_tests):
    pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)
    SE += (pred - rate_test[n, 2])**2 

RMSE = np.sqrt(SE/n_tests)
print ('User-user CF, RMSE =', RMSE)

#--------------------------------------------
# item based
rs = CF(rate_train, k = 30, uuCF = 0)
rs.fit()

joblib.dump(rs, 'iicf')


n_tests = rate_test.shape[0]
SE = 0 # squared error
for n in range(n_tests):
    pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)
    SE += (pred - rate_test[n, 2])**2 

RMSE = np.sqrt(SE/n_tests)
print ('Item-item CF, RMSE =', RMSE)
