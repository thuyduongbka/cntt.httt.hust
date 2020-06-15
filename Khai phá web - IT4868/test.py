import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse 
class CF(object):
    """docstring for CF"""
    def __init__(self, Y_data, k, dist_func = cosine_similarity, uuCF = 1):
        self.uuCF = uuCF # user-user (1) or item-item (0) CF
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        self.k = k
        self.dist_func = dist_func
        self.Ybar_data = None
        #số lượng user và item
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1 
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1
       
    
    def add(self, new_data):        
        self.Y_data = np.concatenate((self.Y_data, new_data), axis = 0)       
    
    def normalize_Y(self):
        users = self.Y_data[:, 0] # lấy ra tất cả các user
        self.Ybar_data = self.Y_data.copy() #đây sẽ là mảng để cập nhật các ratting thành các giá trị chuẩn hóa
        self.mu = np.zeros((self.n_users,)) #mảng lưu các giá trị đánh giá trung bình của user
        
        for n in range(self.n_users):
            
            # Lấy chỉ số đánh giá của của user n
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
            # [ 1.75  0.75 -1.25 -1.25]  
                

        # Đưa về dạng (item,user) = value
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
            (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))

        # các mục trùng lặp được tổng hợp với nhau
        self.Ybar = self.Ybar.tocsr()       

    def similarity(self):
        eps = 1e-6
        # Ybar.T ma trận chuyển vị
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)
        #trả về ma trận độ tương đồng     
    
        
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
                rating = self.__pred(u, i, 1)
                if rating > 2: 
                    recommended_items.append(i)
        
        return recommended_items 
        

    def print_recommendation(self):
        print ('Recommendation: ')
        for u in range(self.n_users):
            recommended_items = self.recommend(u)
            if self.uuCF:
                print ('Recommend item(s):', recommended_items, 'for user', u , "\n")
            else: 
                print ('Recommend item', u, 'for user(s) : ', recommended_items)

# data file 
r_cols = ['user_id', 'item_id', 'rating']
ratings = pd.read_csv('ex.dat', sep = ' ', names = r_cols, encoding='latin-1')
Y_data = ratings.to_numpy()

rs = CF(Y_data, k = 2, uuCF = 0)
rs.fit()

print(rs.pred(0,2,1))
print(rs.recommend(0))