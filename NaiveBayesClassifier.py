import numpy as np
import pandas as pd
import math
import sys
#import sys
from collections import defaultdict





class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.oldProb = {}
        self.likelihood = {}

        self.stats = {}

        self.target = 'label'
        self.categorical_features = {'team_abbreviation_home', 'team_abbreviation_away', 'season_type', 'home_wl_pre5', 'away_wl_pre5'}
        self.numFeatures = {
            'min_avg5', 'fg_pct_home_avg5', 'fg3_pct_home_avg5', 'ft_pct_home_avg5',
            'oreb_home_avg5', 'dreb_home_avg5', 'reb_home_avg5', 'ast_home_avg5',
            'stl_home_avg5', 'blk_home_avg5', 'tov_home_avg5', 'pf_home_avg5', 'pts_home_avg5',
            'fg_pct_away_avg5', 'fg3_pct_away_avg5', 'ft_pct_away_avg5', 'oreb_away_avg5',
            'dreb_away_avg5', 'reb_away_avg5', 'ast_away_avg5', 'stl_away_avg5', 'blk_away_avg5',
            'tov_away_avg5', 'pf_away_avg5', 'pts_away_avg5'
        }

    def fit(self, data):
        self.classes = np.unique(data[self.target])
        
   
        for cls in self.classes:
            class_data = data[data[self.target] == cls] #initialize
            self.oldProb[cls] = len(class_data) / len(data)
            self.likelihood[cls] = {}
            self.stats[cls] = {}

           
            for feature in self.categorical_features:
                self.likelihood[cls][feature] = class_data[feature].value_counts(normalize=True).to_dict()

           
            for feature in self.numFeatures:
                mean = class_data[feature].mean()
                std = class_data[feature].std()
                self.stats[cls][feature] = (mean, std)

    def normalPDFfunc(self, x, mean, std):

        if std == 0:
            std = 1e-6 
        exponent = math.exp(-((x - mean) ** 2) / (2 * std ** 2)) # ** is power

        return (1 / (math.sqrt(2 * math.pi) * std)) * exponent #math.pi is 3.14... it is val of pi 

    def usingInst(self, instance):
        class_probs = {}
        for cls in self.classes:
           
            TotalProbability = math.log(self.oldProb[cls])

            for feature in self.categorical_features:
                Value = instance[feature]
                TotalProbability += math.log(self.likelihood[cls][feature].get(Value, 1e-6))

            for feature in self.numFeatures:
                Value = instance[feature]

                mean, std = self.stats[cls][feature]
                TotalProbability += math.log(self.normalPDFfunc(Value, mean, std))

            class_probs[cls] = TotalProbability

        # Return the class with the highest probability
        return max(class_probs, key =class_probs.get)

    def predict(self, data):
        return data.apply(self.usingInst, axis =1)

def preprocessData(df):
    # win for home/win for away 
    df['home_wl_pre5'] = df['home_wl_pre5'].apply(lambda x: sum(1 if ch ==  'W' else 0 for ch in x))

    df['away_wl_pre5'] = df['away_wl_pre5'].apply(lambda x: sum(1 if ch ==  'W' else 0 for ch in x))
    return df

def main():
   
    train_file = sys.argv[1]
    test_file = sys.argv[2]

  
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    train_data = preprocessData(train_data)
    test_data = preprocessData(test_data)
    X_train = train_data.drop(columns=['label'])
    y_train = train_data['label']

    #apply here
    classifier = NaiveBayesClassifier()
    classifier.fit(train_data)

    if 'label' in test_data.columns:
        test_data = test_data.drop(columns=['label'])
    predictions = classifier.predict(test_data)

    for prediction in predictions:
        print(prediction)

if __name__ == "__main__":

    main()
