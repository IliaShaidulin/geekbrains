import numpy as np


def precision_at_k(recommended_list, bought_list, k=5):
        '''Сколько из топ k рекомендованных товаров пользователь на самом деле купил'''
        try:    
                return np.isin(bought_list,recommended_list[:k]).sum()/k
        except:
                return 0

def recall_at_k(recommended_list, bought_list, k=5):
    '''Сколько из купленных пользователем товаров относятся к топ k рекомендаций'''
    try:
        return np.isin(bought_list,recommended_list[:k]).sum()/len(bought_list)
    except:
        return 0