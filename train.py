import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

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
#
# def split(a, n : int):
#     return a[:n].copy(), a[n:].copy()

if __name__ == '__main__':
    train = pd.read_csv('train_V2.csv')
    test = pd.read_csv('test_V2.csv')

    # Red sa null vrednosti za winPlacePerc
    train.drop(2744604, inplace=True)

    # Pravljenje novih kolona
    train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
    train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
    train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)
    train['maxPlaceNorm'] = train['maxPlace']*((100-train['playersJoined'])/100 + 1)
    train['matchDurationNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
    train['healsandboosts'] = train['heals'] + train['boosts']
    train = pd.get_dummies(train, columns=['matchType'])
    train = drop_outliers(train)
    train = encode(train)

    # Pravljenje novih kolona
    test['playersJoined'] = test.groupby('matchId')['matchId'].transform('count')
    test['killsNorm'] = test['kills'] * ((100 - test['playersJoined']) / 100 + 1)
    test['damageDealtNorm'] = test['damageDealt'] * ((100 - test['playersJoined']) / 100 + 1)
    test['maxPlaceNorm'] = test['maxPlace'] * ((100 - test['playersJoined']) / 100 + 1)
    test['matchDurationNorm'] = test['matchDuration'] * ((100 - test['playersJoined']) / 100 + 1)
    test['healsandboosts'] = test['heals'] + test['boosts']
    test = pd.get_dummies(test, columns=['matchType'])
    test = drop_outliers(test)
    test = encode(test)

    #razlikuju se ids u test i train
    train.drop(columns = ['Id'], inplace=True)
    test.drop(columns=['Id'], inplace=True)

    sample = 50000
    train_sample = train.sample(sample)
    test_sample = test.sample(sample)

    x_train = train_sample.drop(columns=['winPlacePerc'])
    y_train = train_sample['winPlacePerc']

    y_test = test_sample['winPlacePerc']
    x_test = test_sample.drop(columns=['winPlacePerc'])

    # val_perc = 0.12  # % to use for validation set
    # n_valid = int(val_perc * sample)
    # n_trn = len(x_train) - n_valid
    # raw_train, raw_valid = split(train_sample, n_trn)
    # X_train, X_valid = split(x_train, n_trn)
    # y_train1, y_valid = split(y_train, n_trn)

    m1 = RandomForestRegressor(n_estimators=70, min_samples_leaf=3, max_features=0.5,
                               n_jobs=-1)
    m1.fit(x_train, y_train)

    print('Score: ', mean_absolute_error(m1.predict(x_test), y_test))