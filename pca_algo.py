import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator


class pca_algo:
    # Number of components to keep.
    n_components, loaded_data = None, None

    def __init__(self, train=None, test=None, predict=None):
        self.pca = PCA()
        self.train, self.test, self.predict, self.temp_sum = train, test, predict, 0
        if pca_algo.loaded_data is None:
            pca_algo.loaded_data = (np.append(self.train, self.test, axis=0)) if self.train is not None else self.predict
            # if loaded_data is None, the n_components would be necessarily None
            self.use_knee_locator()

    def PCAlgo(self, is_predict_proba: bool):
        loaded = pca_algo.loaded_data
        data = (np.append(loaded, self.train, axis=0)) if is_predict_proba else loaded
        pca = PCA(n_components=pca_algo.n_components)
        # Fit the model with loaded_data and apply the dimensionality reduction on loaded_data.
        data = pca.fit_transform(StandardScaler().fit_transform(data))
        size = len(self.train)
        if is_predict_proba:
            # It's a prediction, process the prediction data after loaded_data
            self.predict = data[len(loaded):]
        else:
            # Split the data between train and test
            self.train, self.test = data[:size], data[size:]

    def use_knee_locator(self):
        pca = PCA()
        # Fit the model with the loaded_data.
        pca.fit(pca_algo.loaded_data)
        # Determine explained variance using explained_variance_ration_ attribute
        exp_var_pca = pca.explained_variance_ratio_

        # x = [1, 2, 3, ..., 87, ...,  692]
        x = list(range(1, len(exp_var_pca)+1))
        # y = [0.456, 0.555, 0.630, ..., 0.953, ..., 0.999, 1.000]
        y = [self.sum(evp) for evp in exp_var_pca]

        # curve=”concave” means kneed will detect knees.
        # interp_method=”polynomial” means then x and y will be fit using 'numpy.polyfit'.
        kl = KneeLocator(x, y, curve="concave", interp_method="polynomial")

        # define n_components as the knee we found
        pca_algo.n_components = kl.knee

        print(f'Knee: {kl.knee}')
        # print(f'{y[kl.knee] * 100: .3f}')

    # y helper
    def sum(self, evp):
        self.temp_sum += evp
        return self.temp_sum

    # for X_train & X_test in data_extractor
    def do_pca(self):
        self.PCAlgo(is_predict_proba=False)

    # for predict_proba in emotion_recognition
    def do_pca_predict_proba(self):
        self.PCAlgo(is_predict_proba=True)



