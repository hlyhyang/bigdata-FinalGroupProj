import pandas as pd
import numpy as np
from collections import Counter
import os
from IPython import embed
from tqdm import tqdm

print('Loading data...')
train = pd.read_csv('~/kkbox/data/train_new.csv')
test = pd.read_csv('~/kkbox/data/test_new.csv')
df = pd.concat([train, test], sort=False)
train = df[0:len(train)].drop('id', axis=1)
test = df[len(train):].drop('target', axis=1)
print('Loading data finished')

# drop useless features or bad features
drop = ['is_featured', 'genre_ids_count', 'artist_count', 'artist_composer', 'artist_composer_lyricist',
        'song_lang_boolean', 'smaller_song', ]
for each in drop:
    if each in train.columns:
        train = train.drop(each, axis=1)
    if each in test.columns:
        test = test.drop(each, axis=1)


# deal with bd's outlier
def age_transfer(age):
    if age == 0:
        new_age = np.random.randint(7, 50)
    elif age <= 7 and age > 0:
        new_age = np.random.randint(12, 25)
    elif age >= 75:
        new_age = np.random.randint(45, 75)
    else:
        new_age = age
    return new_age


train['bd'] = train['bd'].apply(age_transfer).astype(np.int64)
test['bd'] = test['bd'].apply(age_transfer).astype(np.int64)

train['song_year'] = train['song_year'].fillna(train['song_year'].median()).astype(np.int64)
test['song_year'] = test['song_year'].fillna(test['song_year'].median()).astype(np.int64)

train['source_screen_name'] = train['source_screen_name'].fillna('Unknown')
test['source_screen_name'] = test['source_screen_name'].fillna('Unknown')

train['source_type'] = train['source_type'].fillna('Unknown')
test['source_type'] = test['source_type'].fillna('Unknown')

train['source_system_tab'] = train['source_system_tab'].fillna('Unknown')
test['source_system_tab'] = test['source_system_tab'].fillna('Unknown')

train['artist_name'] = train['artist_name'].fillna('Unknown')
test['artist_name'] = test['artist_name'].fillna('Unknown')

train['composer'] = train['composer'].fillna('Unknown')
test['composer'] = test['composer'].fillna('Unknown')

combine = [train, test]
for each in combine:
    each['use'] = 1  # 这个又什么用
    print('start making dics......')
    # Calculate every member’s number of play behaviors
    m_c = dict(each['use'].groupby(each['msno']).sum())
    print('m_c finished')
    # Calculate every member’s number of play behaviors of each artist
    m_a_c = dict(each['use'].groupby([each['msno'], each['artist_name']]).sum())
    print('m_a_c finished')
    # Calculate every member’s number of play behaviors of each source_screen_name
    m_s_c = dict(each['use'].groupby([each['msno'], each['source_screen_name']]).sum())
    print('m_s_c finished')
    # Calculate every member’s number of play behaviors of each source_type
    m_st_c = dict(each['use'].groupby([each['msno'], each['source_type']]).sum())
    print('m_st_c finished')
    # Calculate every member’s number of play behaviors of each composer
    m_c_c = dict(each['use'].groupby([each['msno'], each['composer']]).sum())
    print('m_c_c finished')
    # Calculate every member’s number of play behaviors of each genre_ids
    m_g_c = dict(each['use'].groupby([each['msno'], each['genre_ids']]).sum())
    # Calculate every member’s number of play behaviors of each source_screen_name
    m_a_s_c = dict(each['use'].groupby([each['msno'], each['artist_name'], each['source_type']]).sum())
    print('m_a_s_c finished')

    print('dics making finished')

    m_c_d = []
    m_a_c_d = []
    m_s_c_d = []
    m_st_c_d = []
    m_c_c_d = []
    m_g_c_d = []
    m_a_s_c_d = []

    print('start iterrows......')
    for i, row in tqdm(each.iterrows()):
        try:
            m_c_d.append(m_c[row['msno']])
            tup = (row['msno'], row['artist_name'])
            m_a_c_d.append(m_a_c[tup])
            tup = (row['msno'], row['source_screen_name'])
            m_s_c_d.append(m_s_c[tup])
            tup = (row['msno'], row['source_type'])
            m_st_c_d.append(m_st_c[tup])
            tup = (row['msno'], row['composer'])
            m_c_c_d.append(m_c_c[tup])
            tup = (row['msno'], row['artist_name'], row['source_type'])
            m_a_s_c_d.append(m_a_s_c[tup])
            tup = (row['msno'], row['genre_ids'])
            m_g_c_d.append(m_g_c[tup])
        except:
            embed()
    print('iterrows finished and start adding to df......')

    each['m_c'] = m_c_d
    each['m_a_c'] = m_a_c_d
    each['m_s_c'] = m_s_c_d
    each['m_st_c'] = m_st_c_d
    each['m_c_c'] = m_c_c_d
    each['m_g_c'] = m_g_c_d
    each['m_a_s_c'] = m_a_s_c_d

    print('adding finished and start calculating ratios')
    each['m_a_c_ratio'] = (each['m_a_c'] / each['m_c']) * 100
    each['m_s_c_ratio'] = (each['m_s_c'] / each['m_c']) * 100
    each['m_st_c_ratio'] = (each['m_st_c'] / each['m_c']) * 100
    each['m_c_c_ratio'] = (each['m_c_c'] / each['m_c']) * 100
    each['m_g_c_ratio'] = (each['m_g_c'] / each['m_c']) * 100
    each['m_a_s_c_ratio'] = (each['m_a_s_c'] / each['m_c']) * 100
    print('finished')

train = train.drop('use', axis=1)
test = test.drop('use', axis=1)
train.to_csv('~/kkbox/data/fe_train.csv', index=False)
test.to_csv('~/kkbox/data/fe_test.csv', index=False)
