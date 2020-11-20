import numpy as np
import pandas as pd
from IPython import embed
import os

print('Loading data...')
data_path = '~/kkbox/data'
train = pd.read_csv(os.path.join(data_path, 'train.csv'), dtype={'msno': 'category',
                                                                 'source_system_tab': 'category',
                                                                 'source_screen_name': 'category',
                                                                 'source_type': 'category',
                                                                 'target': np.uint8,
                                                                 'song_id': 'category'})
test = pd.read_csv(os.path.join(data_path, 'test.csv'), dtype={'msno': 'category',
                                                               'source_system_tab': 'category',
                                                               'source_screen_name': 'category',
                                                               'source_type': 'category',
                                                               'song_id': 'category'})
songs = pd.read_csv(os.path.join(data_path, 'songs.csv'), dtype={'genre_ids': 'category',
                                                                 'language': 'category',
                                                                 'artist_name': 'category',
                                                                 'composer': 'category',
                                                                 'lyricist': 'category',
                                                                 'song_id': 'category'})
members = pd.read_csv(os.path.join(data_path, 'members.csv'), dtype={'city': 'category',
                                                                     'bd': np.uint8,
                                                                     'gender': 'category',
                                                                     'registered_via': 'category'})
songs_extra = pd.read_csv(os.path.join(data_path, 'song_extra_info.csv'))
print('Loading data finished.')

# deal with the songs information and merge them together
train = train.merge(songs, on='song_id', how='left')  # 左连接，左侧train取全部，右侧songs取部分
test = test.merge(songs, on='song_id', how='left')

# deal with members registration time and expiration time
members['membership_days'] = pd.to_datetime(members['expiration_date'], format='%Y%m%d').subtract(
    pd.to_datetime(members['registration_init_time'], format='%Y%m%d')).dt.days.astype(int)
members['registration_year'] = pd.to_datetime(members['registration_init_time'], format='%Y%m%d').dt.year
members['registration_month'] = pd.to_datetime(members['registration_init_time'], format='%Y%m%d').dt.month
members['registration_date'] = pd.to_datetime(members['registration_init_time'], format='%Y%m%d').dt.day
members['expiration_year'] = pd.to_datetime(members['expiration_date'], format='%Y%m%d').dt.year
members['expiration_month'] = pd.to_datetime(members['expiration_date'], format='%Y%m%d').dt.month
members['expiration_date'] = pd.to_datetime(members['expiration_date'], format='%Y%m%d').dt.day
members = members.drop(['registration_init_time'], axis=1)


# deal with the published time of a song
def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan


songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
songs_extra.drop(['isrc', 'name'], axis=1,
                 inplace=True)  # bool, default False, If True, do operation inplace and return None.

train = train.merge(members, on='msno', how='left')
test = test.merge(members, on='msno', how='left')

train = train.merge(songs_extra, on='song_id', how='left')
train.song_length.fillna(200000, inplace=True)
train.song_length = train.song_length.astype(np.uint32)
train.song_id = train.song_id.astype('category')

test = test.merge(songs_extra, on='song_id', how='left')
test.song_length.fillna(200000, inplace=True)
test.song_length = test.song_length.astype(np.uint32)
test.song_id = test.song_id.astype('category')

train['genre_ids'] = train['genre_ids'].cat.add_categories(['no_genre_id'])
test['genre_ids'] = test['genre_ids'].cat.add_categories(['no_genre_id'])
train['genre_ids'].fillna('no_genre_id', inplace=True)
test['genre_ids'].fillna('no_genre_id', inplace=True)


# count number of the lyrics writer
def lyricist_count(x):
    if x == 'no_lyricist':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1


train['lyricist'] = train['lyricist'].cat.add_categories(['no_lyricist'])
test['lyricist'] = test['lyricist'].cat.add_categories(['no_lyricist'])
train['lyricist'].fillna('no_lyricist', inplace=True)
test['lyricist'].fillna('no_lyricist', inplace=True)
train['lyricists_count'] = train['lyricist'].apply(lyricist_count).astype(np.int8)  # 训练集添加一列新的feature
test['lyricists_count'] = test['lyricist'].apply(lyricist_count).astype(np.int8)


# count number of the composer
def composer_count(x):
    if x == 'no_composer':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1


train['composer'] = train['composer'].cat.add_categories(['no_composer'])
test['composer'] = test['composer'].cat.add_categories(['no_composer'])
train['composer'].fillna('no_composer', inplace=True)
test['composer'].fillna('no_composer', inplace=True)
train['composer_count'] = train['composer'].apply(composer_count).astype(np.int8)
test['composer_count'] = test['composer'].apply(composer_count).astype(np.int8)

# number of times a song has been played before
_dict_count_song_played_train = {k: v for k, v in train['song_id'].value_counts().iteritems()}
_dict_count_song_played_test = {k: v for k, v in test['song_id'].value_counts().iteritems()}


def count_song_played(x):
    global _dict_count_song_played_train, _dict_count_song_played_test
    try:
        return _dict_count_song_played_train[x]
    except KeyError:
        try:
            return _dict_count_song_played_test[x]
        except KeyError:
            return 0


train['count_song_played'] = train['song_id'].apply(count_song_played).astype(np.int64)
test['count_song_played'] = test['song_id'].apply(count_song_played).astype(np.int64)

# number of times the artist has been played
_dict_count_artist_played_train = {k: v for k, v in train['artist_name'].value_counts().iteritems()}
_dict_count_artist_played_test = {k: v for k, v in test['artist_name'].value_counts().iteritems()}


def count_artist_played(x):
    global _dict_count_artist_played_train, _dict_count_artist_played_test
    try:
        return _dict_count_artist_played_train[x]
    except KeyError:
        try:
            return _dict_count_artist_played_test[x]
        except KeyError:
            return 0


train['count_artist_played'] = train['artist_name'].apply(count_artist_played).fillna(0, inplace=True).astype(np.int64)
test['count_artist_played'] = test['artist_name'].apply(count_artist_played).fillna(0, inplace=True).astype(np.int64)

for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
print('Saving preprocessed data.')
train.to_csv('~/kkbox/data/train_new.csv', index=False)
test.to_csv('~/kkbox/data/test_new.csv', index=False)
print('Finished')
