import pandas as pd
import numpy as np
from os.path import join

# 1. read data

ltable = pd.read_csv(join('data', "ltable.csv"))
rtable = pd.read_csv(join('data', "rtable.csv"))
train = pd.read_csv(join('data', "train.csv"))


# 2. blocking
def pairs2LR(ltable, rtable, candset):
    ltable.index = ltable.id
    rtable.index = rtable.id
    pairs = np.array(candset)
    tpls_l = ltable.loc[pairs[:, 0], :]
    tpls_r = rtable.loc[pairs[:, 1], :]
    tpls_l.columns = [col + "_l" for col in tpls_l.columns]
    tpls_r.columns = [col + "_r" for col in tpls_r.columns]
    tpls_l.reset_index(inplace=True, drop=True)
    tpls_r.reset_index(inplace=True, drop=True)
    LR = pd.concat([tpls_l, tpls_r], axis=1)
    return LR


def block_by_brand(ltable, rtable):
    # ensure brand is str
    ltable["brand"] = ltable["brand"].astype(str)
    rtable["brand"] = rtable["brand"].astype(str)
    ltable["modelno"] = ltable["modelno"].astype(str)
    rtable["modelno"] = rtable["modelno"].astype(str)

    # get all brands
    brands_l = set(ltable["brand"].values)
    brands_r = set(rtable["brand"].values)
    brands = brands_l.union(brands_r)

    # map each brand to left ids and right ids
    brand2ids_l = {b.lower(): [] for b in brands}
    brand2ids_r = {b.lower(): [] for b in brands}

    brand2modelnos_l = {b.lower(): [] for b in brands}
    brand2modelnos_r = {b.lower(): [] for b in brands}

    for i, x in ltable.iterrows():
        brand2ids_l[x["brand"].lower()].append(x["id"])
        brand2modelnos_l[x["brand"].lower()].append(x["modelno"])
    for i, x in rtable.iterrows():
        brand2ids_r[x["brand"].lower()].append(x["id"])
        brand2modelnos_r[x["brand"].lower()].append(x["modelno"])

    # put id pairs that share the same brand in candidate set
    matchset = []
    candset = []
    for brd in brands:
        if brd != "nan":
            l_ids = brand2ids_l[brd]
            r_ids = brand2ids_r[brd]
            l_modelnos = brand2modelnos_l[brd]
            r_modelnos = brand2modelnos_r[brd]
            for i in range(len(l_ids)):
                for j in range(len(r_ids)):
                    l_id = l_ids[i]
                    r_id = r_ids[j]
                    l_modelno = l_modelnos[i]
                    r_modelno = r_modelnos[j]
                    if (l_modelno == r_modelno and l_modelno != "nan"):
                        matchset.append([l_ids[i], r_ids[j]])
                    elif (l_modelno == "nan" or r_modelno == "nan"):
                        candset.append([l_ids[i], r_ids[j]])
        else:
            l_ids = brand2ids_l[brd]
            r_ids = brand2ids_r[brd]
            l_modelnos = brand2modelnos_l[brd]
            r_modelnos = brand2modelnos_r[brd]
            for i in range(len(l_ids)):
                l_id = l_ids[i]
                l_modelno = l_modelnos[i]
                for j, x in rtable.iterrows():
                    r_id = x["id"]
                    r_modelno = x["modelno"]
                    if (l_modelno == r_modelno and l_modelno != "nan"):
                        matchset.append([l_ids[i], r_id])
                    elif (l_modelno == "nan" or r_modelno == "nan"):
                        candset.append([l_ids[i], r_id])
            for i in range(len(r_ids)):
                r_id = r_ids[i]
                r_modelno = r_modelnos[i]
                for j, x in ltable.iterrows():
                    l_id = x["id"]
                    l_modelno = x["modelno"]
                    l_brand = x["brand"]
                    if l_brand != "nan":
                        if (l_modelno == r_modelno and l_modelno != "nan"):
                            matchset.append([l_id, r_ids[i]])
                        elif (l_modelno == "nan" or r_modelno == "nan"):
                            candset.append([l_id, r_ids[i]])
    return (matchset, candset)
# blocking to reduce the number of pairs to be compared
block_tup = block_by_brand(ltable, rtable)
matchset = block_tup[0]
candset = block_tup[1]
print("number of pairs originally", ltable.shape[0] * rtable.shape[0])
print("number of pairs after blocking",len(candset))
candset_df = pairs2LR(ltable, rtable, candset)
matchset_df = pairs2LR(ltable, rtable, matchset)



# 3. Feature engineering
import Levenshtein as lev

def jaccard_similarity(row, attr):
    x = set(row[attr + "_l"].lower().split())
    y = set(row[attr + "_r"].lower().split())
    return len(x.intersection(y)) / max(len(x), len(y))


def levenshtein_distance(row, attr):
    x = row[attr + "_l"].lower()
    y = row[attr + "_r"].lower()
    return lev.distance(x, y)

def feature_engineering(LR):
    LR = LR.astype(str)
    attrs = ["title", "category"]
    features = []
    for attr in attrs:
        j_sim = LR.apply(jaccard_similarity, attr=attr, axis=1)
        l_dist = LR.apply(levenshtein_distance, attr=attr, axis=1)
        features.append(j_sim)
        features.append(l_dist)
    features = np.array(features).T
    return features
candset_features = feature_engineering(candset_df)

# also perform feature engineering to the training set
training_pairs = list(map(tuple, train[["ltable_id", "rtable_id"]].values))
training_df = pairs2LR(ltable, rtable, training_pairs)
training_features = feature_engineering(training_df)
training_label = train.label.values

# 4. Model training and prediction
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight="balanced", random_state=0)
rf.fit(training_features, training_label)
y_pred = rf.predict(candset_features)

# 5. output

matching_pairs = candset_df.loc[y_pred == 1, ["id_l", "id_r"]]
matching_pairs = list(map(tuple, matching_pairs.values))

matching_pairs = block_by_brand(ltable, rtable)[0]
matching_pairs = [tuple(x) for x in matching_pairs]

matching_pairs_in_training = training_df.loc[training_label == 1, ["id_l", "id_r"]]
matching_pairs_in_training = set(list(map(tuple, matching_pairs_in_training.values)))

pred_pairs = [pair for pair in matching_pairs if
              pair not in matching_pairs_in_training]  # remove the matching pairs already in training
pred_pairs = np.array(pred_pairs)
pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"])
pred_df.to_csv("output.csv", index=False)
