from logistic_sgd import load_data
from sklearn.cross_validation import train_test_split
import cPickle, gzip
import numpy as np

class dataPartition(object):
    def __init__(
        self,
        dataset_name = 'model_set.csv',
        output_names = ['train_x.csv', 'train_y.csv', 'valid_x.csv', 'valid_y.csv', 'test_x.csv', 'test_y.csv'],
        target_column = -1,
        stratify = True,
        cv_splits = [.70,.20,.10],
        rand_seed = 8675309,
        col_sample = False,
        obs_sample = False,
        bootstrap_factor = False,
        dataset_enlarge = False,
        model_convert = False
        ):
            self.stratify = stratify
            self.cv_splits = cv_splits
            self.col_sample = col_sample
            self.obs_sample = obs_sample
            self.bootstrap_factor = bootstrap_factor
            self.dataset_enlarge = dataset_enlarge
            self.rand_seed = rand_seed
            self.target_column = target_column

            train_rate = cv_splits[0]
            dataset = np.genfromtxt(dataset_name, delimiter=',')

            idx_IN_columns = [i for i in xrange(np.shape(dataset)[1]) if i not in [target_column]]
            X = dataset[1::,idx_IN_columns]
            print X
            y = dataset[1::,target_column]

            if len(cv_splits) > 1:
                valid_rate = cv_splits[1]
                if stratify:
                    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_rate, stratify=y)
                else:
                    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_rate)


            if len(cv_splits) > 2:
                test_rate = cv_splits[2]/(cv_splits[1]+cv_splits[2])
                if stratify:
                    X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=test_rate, stratify=y_valid)
                else:
                    X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=test_rate)

            np.savetxt('train_x.csv', X_train, delimiter=",")
            np.savetxt('train_y.csv', y_train, delimiter=",")
            np.savetxt('valid_x.csv', X_valid, delimiter=",")
            np.savetxt('valid_y.csv', y_valid, delimiter=",")
            np.savetxt('test_x.csv', X_test, delimiter=",")
            np.savetxt('test_y.csv', y_test, delimiter=",")

class PackageData(object):

    def __init__(
        self,
        transformation = 'linear',
        datasets = ['training', 'validation', 'testing', 'score'],
        dataset_names = ['train_x.csv', 'train_y.csv', 'valid_x.csv', 'valid_y.csv', 'test_x.csv', 'test_y.csv', 'score_x.csv'],
        output_name = 'pickled_model_sets',
        print_check = True
        ):
            if transformation not in ['linear','standardize', None]:
                print 'invalid transformation - linear or standardize'
                exit()

            self.transformation = transformation
            self.datasets = datasets
            self.dataset_names = dataset_names
            self.output_name = output_name
            self.print_check = print_check

    def scale_linear_bycolumn(self, rawpoints, high = 1.0, low = 0.0, scoring_set = None):
        mins = np.min(rawpoints, axis=0)
        maxs = np.max(rawpoints, axis=0)
        rng = maxs - mins

        for index in range(0, len(rng)):
            if rng[index] < 1:
                rng[index] = -1
                maxs[index] = 0

        if scoring_set == None:
            return_set = high - (((high - low) * (maxs - rawpoints)) / rng)
        else:
            return_set = high - (((high - low) * (maxs - scoring_set)) / rng)
        return return_set

    def featureScale(self, rawpoints, scoring_set = None, rng_std=False):
        mean = np.mean(rawpoints, axis=0)
        stddev = np.std(rawpoints, axis=0)
        mins = np.min(rawpoints, axis=0)
        maxs = np.max(rawpoints, axis=0)
        rng = maxs - mins

        print rng

        for index in range(0, len(rng)):
            if rng[index] <= 1:
                mean[index] = 0
                stddev[index] = 1

        if scoring_set == None:
            return_set = (rawpoints - mean) / stddev
        else:
            return_set = (scoring_set - mean) / stddev

        return return_set


    def process(self):

        np.set_printoptions(threshold='nan')

        output_list = []

        if 'training' in self.datasets:
            train_x = np.genfromtxt(self.dataset_names[0], delimiter=',')
            train_y = np.genfromtxt(self.dataset_names[1], delimiter=',').astype(int)

            if self.transformation:
                if self.transformation == 'linear':
                    train_x = self.scale_linear_bycolumn(train_x)

                if self.transformation == 'standardize':
                    train_x = self.featureScale(train_x)

            output_list.append((train_x, train_y))

        if 'validation' in self.datasets:
            valid_x = np.genfromtxt(self.dataset_names[2], delimiter=',')
            valid_y = np.genfromtxt(self.dataset_names[3], delimiter=',').astype(int)

            if self.transformation:
                if self.transformation == 'linear':
                    valid_x = self.scale_linear_bycolumn(valid_x)

                if self.transformation == 'standardize':
                    valid_x = self.featureScale(valid_x)

            output_list.append((valid_x, valid_y))

        if 'testing' in self.datasets:
            test_x = np.genfromtxt(self.dataset_names[4], delimiter=',')
            test_y = np.genfromtxt(self.dataset_names[5], delimiter=',').astype(int)

            if self.transformation:
                if self.transformation == 'linear':
                    test_x = self.scale_linear_bycolumn(test_x)

                if self.transformation == 'standardize':
                    test_x = self.featureScale(test_x)

            output_list.append((test_x, test_y))

        if 'score' in self.datasets:
            score_x = np.genfromtxt(self.dataset_names[6], delimiter=',')[1::]

            if self.transformation:
                if self.transformation == 'linear':
                    score_x = self.scale_linear_bycolumn(score_x)

                if self.transformation == 'standardize':
                    score_x = self.featureScale(score_x)

            output_list.append((score_x))

        if self.print_check:
            print valid_x[0]

            print train_x[0]

            print test_x[0]

            print score_x[0]

        f = gzip.open(self.output_name + '.pkl.gz','wb')
        cPickle.dump(output_list, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

if __name__ == "__main__":
    contest = dataPartition(dataset_name = 'numerai_datasets/numerai_training_data.csv', target_column = 21)
    titanic = PackageData(output_name = 'numerai', transformation = None, dataset_names = ['train_x.csv', 'train_y.csv', 'valid_x.csv', 'valid_y.csv', 'test_x.csv', 'test_y.csv', 'numerai_datasets/numerai_tournament_data.csv'])
    titanic.process()
