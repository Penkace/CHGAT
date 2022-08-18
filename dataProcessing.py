# import pandas as pd
# from sklearn.model_selection import train_test_split

# df_train = pd.read_csv('gd_full_pinyin/train.tsv', sep='\t')
# df_dev =  pd.read_csv('gd_full_pinyin/dev.tsv', sep='\t')
# df_train = pd.concat([df_train, df_dev])
# df_train, df_dev = train_test_split(df_train, test_size=0.05)
# df_train.to_csv('gd_full_pinyin/train_9.tsv', index=False, sep = '\t')
# df_dev.to_csv('gd_full_pinyin/dev_1.tsv', index=False, sep = '\t')

# import pandas as pd
# from pypinyin import pinyin, lazy_pinyin, Style
# df =  pd.read_csv('gd.csv')
# char_list = []
# for i, row in df.iterrows():
#     for j in row['char'].split(' '):
#         if j not in char_list:
#             char_list.append(j)

# char_dict = {}
# for i in char_list:
#     char_dict[i] = lazy_pinyin(i)[0]    
# new_list_train = []
# new_list_dev = []

# train_df = pd.read_csv('gd_full/train.tsv', sep='\t')
# for i, row in train_df.iterrows():
#     new_list_train.append('{}|{}'.format(row['text_a'],' '.join([char_dict[j] for j in row['text_a'].split(' ')])))
#     if i %1000000 == 0:
#         print(i)
# # train_df.to_csv('../new_gd_name_data_pinyin/train.tsv',index = None, sep='\t')
# dev_df = pd.read_csv('gd_full/dev.tsv', sep='\t')
# for i, row in dev_df.iterrows():
#     new_list_dev.append('{}|{}'.format(row['text_a'],' '.join([char_dict[j] for j in row['text_a'].split(' ')])))
#     if i %1000000 == 0:
#         print(i)
# # dev_df.to_csv('../new_gd_name_data_pinyin/dev.tsv',index = None, sep='\t')
# train_df.loc[:,'text_a'] = new_list_train
# dev_df.loc[:,'text_a'] = new_list_dev
# train_df.to_csv('gd_full_pinyin/train.tsv',index = None, sep='\t')
# dev_df.to_csv('gd_full_pinyin/dev.tsv',index = None, sep='\t')

# import pandas as pd
# new_list = []
# df = pd.read_csv('gd_full_pinyin/train.tsv', sep='\t')
# for i, row in df.iterrows():
#     new_list.append(row['text_a'].split('|')[0])
#     if i %1000000 == 0:
#         print(i)
# df.loc[:,'text_a'] = new_list
# print(df.head())
# df.to_csv('gd_full/train.tsv',index = None, sep='\t')
# df = None
# new_list = []
# df = pd.read_csv('gd_full_pinyin/dev.tsv', sep='\t')
# for i, row in df.iterrows():
#     new_list.append(row['text_a'].split('|')[0])
#     if i %1000000 == 0:
#         print(i)
# df.loc[:,'text_a'] = new_list
# print(df.head())
# df.to_csv('gd_full/dev.tsv',index = None, sep='\t')

import pandas as pd

df_train = pd.read_csv('test_pinyin/train.tsv', sep='\t')
all = int(df_train.shape[0] / 10)
for i in range(9):
    df_train.iloc[all*i:all*(i+1)].to_csv('test_pinyin/train_{}.tsv'.format(i), index=False, sep = '\t')
df_train.iloc[all*9:].to_csv('test_pinyin/train_{}.tsv'.format(9), index=False, sep = '\t')