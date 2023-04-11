# %%
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender, bm25_weight, tfidf_weight
from scipy.sparse import csr_matrix


# %%
class MainRecommender:
    
    def __init__(self, data, values):
        self.user_item_matrix = self.prepare_matrix(data,values)
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        

        # Топ покупок каждого юзера
        self.top_purchases  = data.groupby(['user_id', 'item_id']).agg({'quantity' :sum, 'sales_value' : sum})\
                                   .sort_values(['user_id', 'sales_value'], ascending=[True,False]).reset_index()
        
        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id').agg({'quantity' :sum, 'sales_value' : sum}).sort_values('sales_value', ascending=False).reset_index()
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        #self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()
        
    
    @staticmethod
    def prepare_matrix(data, values='quantity'):
        """Готовит user-item матрицу"""
        if values == 'binary':
            user_item_matrix = pd.pivot_table(data,index='user_id', columns='item_id',values='quantity',  aggfunc='count',fill_value=0).astype(float)           
        elif values == 'quantity' or values == 'sales_value':
            user_item_matrix = pd.pivot_table(data,index='user_id', columns='item_id',values=values,  aggfunc='sum',fill_value=0).astype(float)
        else:
            raise ValueError ('Значение должно быть выбрано из binary, quantity, sales_value')
        
        return user_item_matrix
    
    def info(self):
        users = self.user_item_matrix.shape[0]
        items = self.user_item_matrix.shape[1]
        facts = users*items - self.user_item_matrix.isin([0]).sum().sum()
        density = facts/(users*items)
        print(f'В матрице пользователей - {users}, товаров - {items}, покупок- {facts}, плотность матрицы составляет {round(density,4)}')  
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id  
    
    def update_dict(self, user_id):
        """Если появился новый user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1
            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})
    
    def fit(self, n_factors=50, regularization=0.001, iterations=15,  weighted=True, own=False, sim_items=False, items=None, n_items=2, knn=1):
        """Обучает ALS (с взвешиванием и без), модель на основе ранее купленных пользователем товаров и модель наиболее похожих товаров"""
        
        if own:
            self.model = ItemItemRecommender(K=knn, num_threads=-1)
        else:
            self.model = AlternatingLeastSquares(factors=n_factors,regularization=regularization,iterations=iterations,num_threads=-1,random_state=42)
    
        if weighted:
            self.csr_user_item_matrix = csr_matrix(tfidf_weight(self.user_item_matrix.T).T)
        else:
            self.csr_user_item_matrix = csr_matrix(self.user_item_matrix)
            
        self.model.fit(self.csr_user_item_matrix, show_progress=False)
        
        if sim_items:
            items = self.user_item_matrix.columns.values if items == None else items
            items_ids = [self.itemid_to_id[item] for item in items]
            recs= self.model.similar_items(items_ids, N=n_items+1)
            top_rec = [x[1:] for x in recs[0]]
            top_rec = pd.Series([[self.id_to_itemid[j] for j in x] for x in top_rec], index=items, name='похожие товары')
            top_scores = pd.Series([x[1:] for x in recs[1]], index=items, name='score')
            return pd.concat([top_rec, top_scores],1).reset_index().rename(columns={'index' : 'item_id'})

    
    def predict(self, users = None, n=5,  filter_already_liked_items=True):
        """Рекомендуем товары на основе методов ALS и на основе ранее купленных пользователем товаров"""
        users  = self.user_item_matrix.index.values if users == None else users
        for user in users:
            self.update_dict(user)
        users_ids = [self.userid_to_id[user] for user in users]
        ids, scores = self.model.recommend(users_ids , self.csr_user_item_matrix[users_ids], N=n, filter_already_liked_items=filter_already_liked_items)
        
        return  pd.Series([[self.id_to_itemid.get(j,'n/a')for j in x] for x in ids], index=users, name='рекомендации').to_frame() 
        
    
    def get_similar_items_recs(self, users=None, n=5):   
        """Рекомендуем товары, похожие на топ-n купленных юзером товаров"""
        
        users  = self.user_item_matrix.index.values if users == None else users
        fin_result= pd.DataFrame(columns = ['рекомендации'])
        for user in users:
            user_top_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(n)
            item_id = user_top_purchases['item_id'].values.tolist()
            sim_items = self.fit(sim_items=True, items=item_id, n_items=1)
            result  = list(set([x[0] for x in sim_items ['похожие товары']]))
            if len(result) < n:
                for item in self.overall_top_purchases.item_id:
                    if item not in result:
                        result.append(item)
                    if len(result) >= n:
                        break
            fin_result.loc[user] = [result]
        return fin_result
    
    
    def get_similar_items_recs_advanced(self, users=None, k=5, n=5, n_items=5):
        """Рекомендуем k товаров, похожих на топ-n купленных юзером товаров. Для каждого товара из n подбирается n_items похожих товаров"""
        
        users  = self.user_item_matrix.index.values if users == None else users
        fin_result= pd.DataFrame(columns = ['рекомендации'])
        for user in users:
            user_top_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(n)
            item_id = user_top_purchases['item_id'].values.tolist()
            
            sim_items = self.fit(sim_items=True, items=item_id, n_items=n_items)
            
            sim_items_advanced = pd.merge(sim_items ,user_top_purchases[['item_id', 'sales_value']])
            result = pd.DataFrame()
            sim_items_advanced ['weighted_score'] = ''
            for i, row in sim_items_advanced.iterrows():
                sim_items_advanced.at[i, 'weighted_score'] = [score*sim_items_advanced.loc[i, 'sales_value'] for score in sim_items_advanced.loc[i,'score']]
                temp = pd.DataFrame(sim_items_advanced.loc[i, 'weighted_score'], index = sim_items_advanced.loc[i, 'похожие товары'],columns = [sim_items_advanced.loc[i, 'item_id']])
                result = pd.concat([result,temp],1).fillna(0)
            result['total_score'] = result.sum(axis=1)
            result = result.sort_values('total_score', ascending=False).head(k).index.to_list()
            fin_result.loc[user] = [result]
        return fin_result

    def get_similar_users_recs(self, users=None, n=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        
        users  = self.user_item_matrix.index.values if users == None else users
        fin_result= pd.DataFrame(columns = ['рекомендации'])
        for user in users:
            self.fit()
            similar_users = self.model.similar_users(self.userid_to_id[user], N=n+1)
            users = [self.id_to_userid[x] for x in similar_users[0][1:]]
            self.fit(own=True)
            preds = self.predict(users=users, n=n)
            result  = list(set([x[0] for x in preds ['рекомендации']]))
            if len(result) < n:   
                for item in self.overall_top_purchases.item_id:
                    if item not in result:
                        result.append(item)
                    if len(result) >= n:
                        break
            fin_result.loc[user] = [result]
        return fin_result
    
    



# %%



