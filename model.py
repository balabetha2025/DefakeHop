import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
import utils
from defakeHop import DefakeHop

class Ensemble():
    def __init__(self, regions=['left_eye', 'right_eye', 'mouth'], num_frames=6, verbose=True):
        self.regions = regions
        self.num_frames = num_frames
        self.defakeHop = {}
        self.classifier = None
        self.features = {}
        self.labels = None
        self.names = None
        self.predict_features = {}
        self.predict_names = None
        self.verbose = verbose

    def fit_region(self, region, images, labels, names, multi_cwSaab_parm):
        if self.verbose:
            print("==============================" + region + "==============================")
        if len(self.defakeHop) == 0:
            self.labels = labels
            self.names = names
        else:
            assert np.array_equal(names, self.names)
            assert np.array_equal(labels, self.labels)
        defakehop = DefakeHop(**multi_cwSaab_parm)
        features = defakehop.fit(images, labels)
        self.defakeHop[region] = defakehop
        self.features[region] = features
        return self

    def predict_region(self, region, images, names):
        if self.verbose:
            print("==============================" + region + "==============================")
        if self.predict_names is None:
            self.predict_names = names
        else:
            assert np.array_equal(names, self.predict_names)
        self.predict_features[region] = self.defakeHop[region].predict(images)
    
    def clean_buffer(self):
        self.features = {}
        self.labels = None
        self.names = None
        self.predict_features = {}
        self.predict_names = None
        
    def concatenate_features(self, regions=None, train_flag=True):
        if self.verbose:
            print("===============Concatenation===============")
        if regions is None:
            regions = self.regions
        features = self.concatenate_regions_features(regions,train_flag=train_flag)
        names = self.names if train_flag else self.predict_names
        return self.concatenate_frames_features(features, self.labels, names, train_flag=train_flag)

    def concatenate_regions_features(self, regions, train_flag):
        feats = []
        for region in regions:
            if train_flag:
                feats.extend(self.features[region].T.tolist())
            else:
                feats.extend(self.predict_features[region].T.tolist())
        return np.array(feats).T

    def concatenate_frames_features(self, features, labels, names, train_flag):
        frames_labels = []
        frames_names = []
        all_frames_features = []
        for idx, prob in enumerate(features):
            cur_vid_name = utils.vid_name(names[idx])
            cur_frame = utils.frame(names[idx])
            frames_features = []
            for i in range(self.num_frames):
                if idx+i < len(names) and utils.vid_name(names[idx+i]) == cur_vid_name and utils.frame(names[idx+i]) == cur_frame + i*6:
                    frames_features.extend(features[idx+i])
                else:
                    break
            if len(frames_features) == self.num_frames*len(prob):
                all_frames_features.append(frames_features)
                if train_flag:
                    frames_labels.append(labels[idx])
                frames_names.append(names[idx])
        if train_flag:
            frames_labels = np.array(frames_labels)
        frames_names = np.array(frames_names)
        all_frames_features = np.array(all_frames_features)
        return all_frames_features, frames_labels, frames_names

    def train_classifier(self, folds=4, param_comb=20, clean=True):
        features, labels, names = self.concatenate_features()
        if self.verbose:
            print("===============Training Classifier===============")
            print("Features shape:", features.shape)
        params = {
        'min_child_weight': [1, 2, 3, 5, 7, 11, 13, 17, 19, 23],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1],
        'colsample_bytree': [0.6, 0.8, 1],
        'max_depth': [6]
        }
        labels = labels.astype(int)
        xgb = XGBClassifier(n_estimators=100, learning_rate=0.2, eval_metric='auc',
                            objective='binary:logistic', tree_method='hist',
                            predictor='cpu_predictor',
                            scale_pos_weight=(len(labels[labels==0])/len(labels[labels==1])),
                            use_label_encoder=False)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        clf = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb,
                                 scoring='roc_auc', n_jobs=8, cv=skf.split(features,labels),
                                 random_state=1001)
        clf.fit(features, labels)
        self.classifier = clf.best_estimator_
        prob = clf.predict_proba(features)[:,1]
        if clean:
            self.clean_buffer()
        return prob, names

    def predict_classifier(self, clean=True):
        features, _, names = self.concatenate_features(train_flag=False)

        if self.verbose:
            print("===============Prediction===============")
            print("Features shape:", features.shape)

        # ✅ FIX HERE — Ensure correct shape for XGBoost
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        else:
            features = features.reshape(features.shape[0], -1)

        prob = self.classifier.predict_proba(features)[:,1]

        if clean:
            self.clean_buffer()
        return prob, names
