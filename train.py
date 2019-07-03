import pandas as pd
import sklearn
from sklearn import metrics
from scipy.cluster import hierarchy as hc
from fastai.imports import *
import os
from sklearn_pandas import DataFrameMapper
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.ensemble import forest

def drop_outliers(dataset):
    # Create feature totalDistance
    dataset['totalDistance'] = dataset['rideDistance'] + dataset['walkDistance'] + dataset['swimDistance']
    # Create feature killsWithoutMoving
    dataset['killsWithoutMoving'] = ((dataset['kills'] > 0) & (dataset['totalDistance'] == 0))

    # Create headshot_rate feature
    dataset['headshot_rate'] = dataset['headshotKills'] / dataset['kills']
    dataset['headshot_rate'] = dataset['headshot_rate'].fillna(0)

    # Remove outliers
    dataset.drop(dataset[dataset['killsWithoutMoving'] == True].index, inplace=True)

    # Drop roadKill 'cheaters'
    dataset.drop(dataset[dataset['roadKills'] > 10].index, inplace=True)

    # More than 30 kills, possible cheaters
    dataset.drop(dataset[dataset['kills'] > 30].index, inplace=True)

    # Huge distance kills
    dataset.drop(dataset[dataset['longestKill'] >= 1000].index, inplace=True)

    # Huge walking distance
    dataset.drop(dataset[dataset['walkDistance'] >= 10000].index, inplace=True)

    # Remove ride distance outliers
    dataset.drop(dataset[dataset['rideDistance'] >= 20000].index, inplace=True)

    # Players who swam more than 2 km
    dataset.drop(dataset[dataset['swimDistance'] >= 2000].index, inplace=True)

    # Players who acquired more than 80 weapons
    dataset.drop(dataset[dataset['weaponsAcquired'] >= 80].index, inplace=True)

    # 40 or more healing items used
    dataset.drop(dataset[dataset['heals'] >= 40].index, inplace=True)
    return dataset

def encode(dataset):
    # Turn groupId and match Id into categorical types
    dataset['groupId'] = dataset['groupId'].astype('category')
    dataset['matchId'] = dataset['matchId'].astype('category')

    # Get category coding for groupId and matchID
    dataset['groupId_cat'] = dataset['groupId'].cat.codes
    dataset['matchId_cat'] = dataset['matchId'].cat.codes

    # Get rid of old columns
    dataset.drop(columns=['groupId', 'matchId'], inplace=True)

    # Lets take a look at our newly created features
    dataset[['groupId_cat', 'matchId_cat']].head()

    return dataset

def proc_df(df, y_fld, skip_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    df = df.copy()
    if preproc_fn: preproc_fn(df)
    y = df[y_fld].values
    df.drop(skip_flds+[y_fld], axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    res = [pd.get_dummies(df, dummy_na=True), y, na_dict]
    if do_scale: res = res + [mapper]
    return res

def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)

def set_rf_samples(n):
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))

def reset_rf_samples():
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))


if __name__ == '__main__':
    train = pd.read_csv(os.join_dir('dataset', 'train_V2.csv'))
    test = pd.read_csv(os.join_dir('dataset', 'test_V2.csv'))

    # Red sa null vrednosti za winPlacePerc
    train.drop(2744604, inplace=True)

    # Pravljenje novih kolona
    train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
    train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
    train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)
    train['maxPlaceNorm'] = train['maxPlace']*((100-train['playersJoined'])/100 + 1)
    train['matchDurationNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
    train['healsandboosts'] = train['heals'] + train['boosts']

    train = drop_outliers(train)
    train = encode(train)

    # Drop Id column, because it probably won't be useful for our Machine Learning algorithm,
    # because the test set contains different Id's
    train.drop(columns = ['Id'], inplace=True)

    print(train)