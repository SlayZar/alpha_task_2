import pandas as pd
import numpy as np
import joblib
import torch.nn as nn
import torch
import pickle
import os
from alpha_task_2.dataset_preprocessing_utils import transform_transactions_to_sequences, create_padded_buckets
from scipy import stats
from tqdm.notebook import tqdm
from catboost import CatBoostClassifier, Pool
from alpha_task_2.utils import read_parquet_dataset_from_local
from alpha_task_2.pytorch_training import inference
from alpha_task_2.data_generators import batches_generator, transaction_features

with open('alpha_task_2/constants/embedding_projections.pkl', 'rb') as f:
    embedding_projections = pickle.load(f)
with open('alpha_task_2/constants/dense_features_buckets.pkl', 'rb') as f:
    dense_features_buckets = pickle.load(f)
with open('alpha_task_2/constants/buckets_info.pkl', 'rb') as f:
    mapping_seq_len_to_padded_len = pickle.load(f)

# Софтмакс
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Агрегаты для бустинга через pivot    
def __amnt_pivot_table_by_column_as_frame(frame, column, agg_funcs=None) -> pd.DataFrame:
    if agg_funcs is None:
        agg_funcs = ['mean', 'count']
    aggs = pd.pivot_table(frame, values='amnt',
                          index=['app_id'], columns=[column],
                          aggfunc={'amnt': agg_funcs},
                          fill_value=0.0)
    aggs.columns = [f'{col[0]}_{column}_{col[1]}' for col in aggs.columns.values]
    return aggs


# Функция для расчёта фичей для модели бустинга
def preprocess(train_df):
  good_cols = joblib.load('alpha_task_2/models/good_cols')
  train_df['amnt_rub'] = np.exp(train_df['amnt']*17.8209)-1
  cols1 = ['amnt', 'amnt_rub', 'hour_diff', 'days_before']
  feat0 = pd.DataFrame()
  for i in cols1:
    if i == 'amnt' or i == 'amnt_rub':
      amnt_feat = pd.DataFrame(train_df.groupby(['app_id'])[i].agg(['count', 'mean', 'max', 'min', 'sum', 'median', 'std']))
      amnt_feat = amnt_feat.merge(pd.DataFrame(train_df[train_df.amnt==0].groupby(['app_id'])[i].agg(['count'])), on = 'app_id', how='left')
      amnt_feat.columns=[f'{i}_count', f'{i}_mean', f'{i}_max', f'{i}_min', f'{i}_sum', f'{i}_zero_count', f'{i}_median', f'{i}_std']
      amnt_feat[f'{i}_zero_share'] =  amnt_feat[f'{i}_zero_count'] / amnt_feat[f'{i}_count'] 
    else:
      amnt_feat = pd.DataFrame(train_df.groupby(['app_id'])[i].agg(['mean', 'max', 'min', 'sum', 'median', 'std']))
      amnt_feat.columns=[ f'{i}_mean', f'{i}_max', f'{i}_min', f'{i}_sum', f'{i}_median', f'{i}_std']
    feat0 = pd.concat([amnt_feat, feat0], axis=1)

  cols2 = ['currency', 'operation_kind', 'card_type', 'operation_type', 'operation_type_group',
           'ecommerce_flag',	'payment_system',  'mcc',
           		'city', 'mcc_category', 'day_of_week', 'hour', 'weekofyear']
  feat = pd.DataFrame() 
  for i in cols2:
    if i in ['card_type', 'mcc', 'mcc_category', 'hour', 'weekofyear']:
      currency_feat = pd.DataFrame(train_df.groupby(['app_id'])[i]\
                              .agg([lambda x: x.nunique(), lambda x: stats.mode(x)[0][0],
                                    lambda x: stats.mode(x)[1][0] / x.count()]))
      currency_feat.columns=[f'{i}_nunique', f'{i}_most_pop', f'{i}_most_pop_share']
    else:
      currency_feat = pd.DataFrame(train_df.groupby(['app_id'])[i]\
                              .agg([lambda x: stats.mode(x)[0][0],
                                    lambda x: stats.mode(x)[1][0] / x.count()]))
      currency_feat.columns=[f'{i}_most_pop', f'{i}_most_pop_share']
    feat = pd.concat([currency_feat, feat], axis=1)
      
  pivot_tables = []
  for col in cols2[:-1]:
      pivot_tables.append(__amnt_pivot_table_by_column_as_frame(train_df, column=col))
  pivot_tables = pd.concat(pivot_tables, axis=1)
  all_feat = feat.merge(feat0, left_index=True, right_index=True).merge(pivot_tables, left_index=True, right_index=True)
  our_cols = []
  for i in good_cols:
    if i in all_feat.columns:
      our_cols.append(i)
  for i in all_feat.columns:
    if 'rub' in i:
      our_cols.append(i)
  return all_feat[our_cols]
  
  
# Функция для скоринга моделями бустинга и сохранения результатов
def boost_scor(t_s, sample_subm, model_paths, sol_path, model_number):
  cols_path = f'{model_paths}/cols_{model_number}'
  model_path = f'{model_paths}/model_{model_number}'
  cats_path = f'{model_paths}/cats_{model_number}'
  output_path = f'{sol_path}/sol_{model_number}.csv'
  test_scores = t_s.copy()
  cols = joblib.load(cols_path)
  cb2 = CatBoostClassifier()
  cb2.load_model(model_path)
  cats = joblib.load(cats_path)
  test_pool = Pool(test_scores[cols], cat_features = cats)
  test_scores['score'] = cb2.predict_proba(test_pool)[:,1]
  test_scores = test_scores[['app_id', 'score']]
  sample_subm2 = sample_subm.merge(test_scores, on=['app_id']).drop(['product'], axis=1)
  sample_subm2.rename(columns={'score': 'flag'}, inplace=True)
  sample_subm2.to_csv(output_path, index=False)
  

# Функция для подготовки данных для нейросети
def create_buckets_from_transactions(path_to_dataset, save_to_path, frame_with_ids = None, 
                                     num_parts_to_preprocess_at_once: int = 1, 
                                     num_parts_total=50, has_target=False):
    block = 0
    for step in tqdm(range(0, num_parts_total, num_parts_to_preprocess_at_once), 
                                   desc="Transforming transactions data"):
        transactions_frame = read_parquet_dataset_from_local(path_to_dataset, step, num_parts_to_preprocess_at_once, 
                                                             verbose=True)
        for dense_col in ['amnt', 'days_before', 'hour_diff']:
            transactions_frame[dense_col] = np.digitize(transactions_frame[dense_col], bins=dense_features_buckets[dense_col])
            
        seq = transform_transactions_to_sequences(transactions_frame)
        seq['sequence_length'] = seq.sequences.apply(lambda x: len(x[1]))
        
        if frame_with_ids is not None:
            seq = seq.merge(frame_with_ids, on='app_id')

        block_as_str = str(block)
        if len(block_as_str) == 1:
            block_as_str = '00' + block_as_str
        else:
            block_as_str = '0' + block_as_str
            
        processed_fragment =  create_padded_buckets(seq, mapping_seq_len_to_padded_len, has_target=has_target, 
                                                    save_to_file_path=os.path.join(save_to_path, 
                                                                                   f'processed_chunk_{block_as_str}.pkl'))
        block += 1
		
# Функция для скоринга бустинга    
def boost_scor(t_s, sample_subm, model_paths, sol_path, model_number):
  cols_path = f'{model_paths}/cols_{model_number}'
  model_path = f'{model_paths}/model_{model_number}'
  cats_path = f'{model_paths}/cats_{model_number}'
  output_path = f'{sol_path}/sol_{model_number}.csv'
  test_scores = t_s.copy()
  cols = joblib.load(cols_path)
  cb2 = CatBoostClassifier()
  cb2.load_model(model_path)
  cats = joblib.load(cats_path)
  test_pool = Pool(test_scores[cols], cat_features = cats)
  test_scores['score'] = cb2.predict_proba(test_pool)[:,1]
  test_scores = test_scores[['app_id', 'score']]
  sample_subm2 = sample_subm.merge(test_scores, on=['app_id'])
  sample_subm2.rename(columns={'score': 'flag'}, inplace=True)
  sample_subm2.to_csv(output_path, index=False)

# Функция для скоринга моделью нейросети
def nn_scoring(path_to_test_dataset, path_to_checkpoints, model_name, result_path, device):
  dir_with_test_datasets = os.listdir(path_to_test_dataset)
  dataset_test = sorted([os.path.join(path_to_test_dataset, x) for x in dir_with_test_datasets])
  if 'nn_model_3' in model_name:
    model = TransactionsRnn2(transaction_features, embedding_projections, top_classifier_units=64).to(device)
  else:
    model = TransactionsRnn(transaction_features, embedding_projections, top_classifier_units=128).to(device)
  model.load_state_dict(torch.load(os.path.join(path_to_checkpoints, model_name)))
  test_preds = inference(model, dataset_test, batch_size=128, device=device)
  test_preds.to_csv(result_path, index=None)

# Агрегация скорингов
def agg_scoring(sols_path, models_path):
  s9 = pd.read_csv(f'{sols_path}/sol_9.csv', names = ['app_id', 'sol_9'], skiprows=1)
  s10 = pd.read_csv(f'{sols_path}/sol_10.csv', names = ['app_id', 'sol_10'], skiprows=1)
  s12 = pd.read_csv(f'{sols_path}/sol_12.csv', names = ['app_id', 'sol_12'], skiprows=1)
  s13 = pd.read_csv(f'{sols_path}/sol_13.csv', names = ['app_id', 'sol_13'], skiprows=1)
  rnn1 = pd.read_csv(f'{sols_path}/nn_mod1.csv', names = ['app_id', 'score_1'], skiprows=1)
  rnn2 = pd.read_csv(f'{sols_path}/nn_mod2.csv', names = ['app_id', 'score_2'], skiprows=1)
  rnn3 = pd.read_csv(f'{sols_path}/nn_mod3.csv', names = ['app_id', 'score_3'], skiprows=1)

  s_m = s9.merge(s10, on=['app_id']).merge(s12, on=['app_id']).merge(s13, on=['app_id'])\
    .merge(rnn1, on=['app_id']).merge(rnn2, on=['app_id']).merge(rnn3, on=['app_id'])
  
  for i in range(3):
    s_m[f'nn_{i+1}_soft'] = softmax(s_m[f'score_{i+1}'])
    s_m[f'nn_{i+1}_rank'] = s_m[f'nn_{i+1}_soft'].rank()
  for i in [9,10,12,13]:
      s_m[f'sol_{i}_soft'] = softmax(s_m[f'sol_{i}'])
      s_m[f'sol_{i}_rank'] = s_m[f'sol_{i}_soft'].rank()

  agg_cols = joblib.load(f'{models_path}/agg_cols')
  cb = CatBoostClassifier()
  cb.load_model(f'{models_path}/agg_model')
  s_m['flag'] = cb.predict_proba(s_m[agg_cols])[:,1]
  s_m[['app_id', 'flag']].to_csv(f'{sols_path}/final_solution.csv', index=False)
	
# Модель нейросети
class TransactionsRnn(nn.Module):
    def __init__(self, transactions_cat_features, embedding_projections, product_col_name='product', 
                 rnn_units=128, top_classifier_units=32):
        super(TransactionsRnn, self).__init__()
        
        self._transaction_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature], 
                                                                                            padding_idx=None) 
                                                          for feature in transactions_cat_features])
        self._spatial_dropout = nn.Dropout2d(0.05)
        self._transaction_cat_embeddings_concated_dim = sum([embedding_projections[x][1] for x in transactions_cat_features])
        
        self._product_embedding = self._create_embedding_projection(*embedding_projections[product_col_name], padding_idx=None)
        
        self._gru = nn.GRU(input_size=self._transaction_cat_embeddings_concated_dim,
                             hidden_size=rnn_units, batch_first=True, bidirectional=True)
        
        self._hidden_size = rnn_units
        
        # построим классификатор, он будет принимать на вход: 
        # [max_pool(gru_states), avg_pool(gru_states), product_embed]
        pooling_result_dimension = self._hidden_size * 2
         
        self._top_classifier = nn.Sequential(nn.Linear(in_features=2*pooling_result_dimension + 
                                                       embedding_projections[product_col_name][1], 
                                                       out_features=top_classifier_units),
                                             nn.ReLU(),
                                             nn.Linear(in_features=top_classifier_units, out_features=1)
                                            )
        
    def forward(self, transactions_cat_features, product_feature):
        batch_size = product_feature.shape[0]
        
        embeddings = [embedding(transactions_cat_features[i]) for i, embedding in enumerate(self._transaction_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)
        concated_embeddings = concated_embeddings.permute(0, 2, 1).unsqueeze(3)
        
        dropout_embeddings = self._spatial_dropout(concated_embeddings)
        dropout_embeddings = dropout_embeddings.squeeze(3).permute(0, 2, 1)

        states, _ = self._gru(dropout_embeddings)
        
        rnn_max_pool = states.max(dim=1)[0]
        rnn_avg_pool = states.sum(dim=1) / states.shape[1]        
        
        product_embed = self._product_embedding(product_feature)
                
        combined_input = torch.cat([rnn_max_pool, rnn_avg_pool, product_embed], dim=-1)
            
        logit = self._top_classifier(combined_input)        
        return logit
    
    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size, padding_idx=padding_idx)


class TransactionsRnn2(nn.Module):
    def __init__(self, transactions_cat_features, embedding_projections, product_col_name='product', 
                 rnn_units=128, top_classifier_units=32):
        super(TransactionsRnn2, self).__init__()
        
        self._transaction_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature], 
                                                                                            padding_idx=None) 
                                                          for feature in transactions_cat_features])
        self._spatial_dropout = nn.Dropout2d(0.1)
        self._transaction_cat_embeddings_concated_dim = sum([embedding_projections[x][1] for x in transactions_cat_features])
        
        self._product_embedding = self._create_embedding_projection(*embedding_projections[product_col_name], padding_idx=None)
        
        self._gru = nn.GRU(input_size=self._transaction_cat_embeddings_concated_dim,
                             hidden_size=rnn_units, batch_first=True, bidirectional=True)
        
        self._hidden_size = rnn_units
        
        # построим классификатор, он будет принимать на вход: 
        # [max_pool(gru_states), avg_pool(gru_states), product_embed]
        pooling_result_dimension = self._hidden_size * 2
         
        self._top_classifier = nn.Sequential(nn.Linear(in_features=2*pooling_result_dimension + 
                                                       embedding_projections[product_col_name][1], 
                                                       out_features=top_classifier_units),
                                             nn.ReLU(),
                                             nn.Linear(in_features=top_classifier_units, out_features=1)
                                            )
        
    def forward(self, transactions_cat_features, product_feature):
        batch_size = product_feature.shape[0]
        
        embeddings = [embedding(transactions_cat_features[i]) for i, embedding in enumerate(self._transaction_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)
        concated_embeddings = concated_embeddings.permute(0, 2, 1).unsqueeze(3)
        
        dropout_embeddings = self._spatial_dropout(concated_embeddings)
        dropout_embeddings = dropout_embeddings.squeeze(3).permute(0, 2, 1)

        states, _ = self._gru(dropout_embeddings)
        
        rnn_max_pool = states.max(dim=1)[0]
        rnn_avg_pool = states.sum(dim=1) / states.shape[1]        
        
        product_embed = self._product_embedding(product_feature)
                
        combined_input = torch.cat([rnn_max_pool, rnn_avg_pool, product_embed], dim=-1)
            
        logit = self._top_classifier(combined_input)        
        return logit
    
    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size, padding_idx=padding_idx)
