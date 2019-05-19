import os
import pandas as pd
import numpy as np
import pickle
import shap
import eli5
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
                            recall_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from tpot import TPOTClassifier
from defragTrees import DefragModel


"""
### UNO EL DATASET ORIGINAL CON LAS FEATURES CREADAS POR EL PIPELINE ###
X_transformed_denormalized = X.values
for trf_feat in range(len(list(X_transformed[0,:]))-1,len(list(X))-1,-1):
    X_transformed_denormalized = np.append(X_transformed_denormalized,
                                            X_transformed[:,[trf_feat]],
                                            axis=1)
"""


class AutoEDM():

    def __init__(self, dataset_path, target, model_path,
                 show_feature_importances=False):

        # CONFIGURACION DE ALGORITMOS A UTILIZAR
        self.classifier_config_dict = {

            # Classifiers

            'sklearn.tree.DecisionTreeClassifier': {
                'criterion': ["gini", "entropy"],
                'max_depth': range(1, 11),
                'min_samples_split': range(2, 21),
                'min_samples_leaf': range(1, 21)
            },

            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ["gini", "entropy"],
                'max_features': np.arange(0.05, 1.01, 0.05),
                'min_samples_split': range(2, 21),
                'min_samples_leaf': range(1, 21),
                'bootstrap': [True, False]
            },

            'sklearn.ensemble.RandomForestClassifier': {
                'n_estimators': [100],
                'criterion': ["gini", "entropy"],
                'max_features': np.arange(0.05, 1.01, 0.05),
                'min_samples_split': range(2, 21),
                'min_samples_leaf':  range(1, 21),
                'bootstrap': [True, False]
            },

            'sklearn.ensemble.GradientBoostingClassifier': {
                'n_estimators': [100],
                'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
                'max_depth': range(1, 11),
                'min_samples_split': range(2, 21),
                'min_samples_leaf': range(1, 21),
                'subsample': np.arange(0.05, 1.01, 0.05),
                'max_features': np.arange(0.05, 1.01, 0.05)
            },

            'xgboost.XGBClassifier': {
                'n_estimators': [100],
                'max_depth': range(1, 11),
                'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
                'subsample': np.arange(0.05, 1.01, 0.05),
                'min_child_weight': range(1, 21),
                'nthread': [1]
            },

            # Preprocesssors
            'sklearn.preprocessing.Binarizer': {
                'threshold': np.arange(0.0, 1.01, 0.05)
            },

            'sklearn.decomposition.FastICA': {
                'tol': np.arange(0.0, 1.01, 0.05)
            },

            'sklearn.cluster.FeatureAgglomeration': {
                'linkage': ['ward', 'complete', 'average'],
                'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
            },

            'sklearn.preprocessing.MaxAbsScaler': {
            },

            'sklearn.preprocessing.MinMaxScaler': {
            },

            'sklearn.preprocessing.Normalizer': {
                'norm': ['l1', 'l2', 'max']
            },

            'sklearn.kernel_approximation.Nystroem': {
                'kernel': ['rbf', 'cosine', 'chi2', 'laplacian',
                           'polynomial', 'poly', 'linear', 'additive_chi2',
                           'sigmoid'],
                'gamma': np.arange(0.0, 1.01, 0.05),
                'n_components': range(1, 11)
            },

            'sklearn.decomposition.PCA': {
                'svd_solver': ['randomized'],
                'iterated_power': range(1, 11)
            },

            'sklearn.preprocessing.PolynomialFeatures': {
                'degree': [2],
                'include_bias': [False],
                'interaction_only': [False]
            },

            'sklearn.kernel_approximation.RBFSampler': {
                'gamma': np.arange(0.0, 1.01, 0.05)
            },

            'sklearn.preprocessing.RobustScaler': {
            },

            'sklearn.preprocessing.StandardScaler': {
            },

            'tpot.builtins.ZeroCount': {
            },

            'tpot.builtins.OneHotEncoder': {
                'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
                'sparse': [False],
                'threshold': [10]
            },

            # Selectors
            'sklearn.feature_selection.SelectFwe': {
                'alpha': np.arange(0, 0.05, 0.001),
                'score_func': {
                    'sklearn.feature_selection.f_classif': None
                }
            },

            'sklearn.feature_selection.SelectPercentile': {
                'percentile': range(1, 100),
                'score_func': {
                    'sklearn.feature_selection.f_classif': None
                }
            },

            'sklearn.feature_selection.VarianceThreshold': {
                'threshold': [0.0001, 0.0005, 0.001,
                              0.005, 0.01, 0.05, 0.1, 0.2]
            },

            'sklearn.feature_selection.RFE': {
                'step': np.arange(0.05, 1.01, 0.05),
                'estimator': {
                    'sklearn.ensemble.ExtraTreesClassifier': {
                        'n_estimators': [100],
                        'criterion': ['gini', 'entropy'],
                        'max_features': np.arange(0.05, 1.01, 0.05)
                    }
                }
            },

            'sklearn.feature_selection.SelectFromModel': {
                'threshold': np.arange(0, 1.01, 0.05),
                'estimator': {
                    'sklearn.ensemble.ExtraTreesClassifier': {
                        'n_estimators': [100],
                        'criterion': ['gini', 'entropy'],
                        'max_features': np.arange(0.05, 1.01, 0.05)
                    }
                }
            }

        }

        self.target = target
        self.model_path = model_path

        # CARGA DE DATASET
        self.df = self._createDataset(dataset_path)
        self.df = self._manageNulls(self.df)

        # MINIMO PRE-PROCESAMIENTO PARA QUE FUNCIONE EL ALGORITMO DE AUTOML
        self.X, self.y = self._preProcessData(self.df, self.target)

        if(show_feature_importances):
            print('-------- EARLY FEATURE IMPORTANCES --------')
            rf = RandomForestClassifier()
            rf.fit(self.X, self.y)
            feature_importances = pd.DataFrame(rf.feature_importances_,
                                               index=self.X.columns,
                                               columns=['importance']
                                               ).sort_values('importance',
                                                             ascending=False)
            print(feature_importances)
            print()

    def loadModel(self):
        # SEPARO DATASETS DE ENTRENAMIENTO Y TEST
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            train_size=0.75,
                                                            test_size=0.25,
                                                            random_state=22)

        # CARGO UN MODELO YA ENTRENADO O ENTRENO UNO NUEVO
        self.pipeline = self._optimizeModel(X_train,
                                            y_train,
                                            self.model_path,
                                            self.classifier_config_dict)
        self._trf_pipeline, self._model_used = self._separatePipelines()
        self._feature_names = list(self.X)
        X_transformed = self._trf_pipeline.fit_transform(self.X)
        for n_feat in range(len(list(X_transformed[0, :]))-len(list(self.X))):
            self._feature_names.append(f'Xt{n_feat}')

    def _preProcessData(self, df, target=None):
        if target is not None:
            X = df.drop(target, axis=1)
            y = df[[target]].values.ravel()
            if isinstance(y[0], str):
                label_encoder = LabelEncoder()
                label_encoder = label_encoder.fit(y)
                y = label_encoder.transform(y)
        else:
            y = None
            X = df
        X = pd.get_dummies(X)
        return (X, y)

    def _manageNulls(self, data):
        newData = data.copy()
        null_columns = newData.columns[newData.isnull().any()]
        if(not null_columns.empty):
            print("WARNING: {} columns contain some null\
                content".format(null_columns))
            impNumeric = SimpleImputer(strategy='mean')
            impCategorical = SimpleImputer(strategy='most_frequent')
            for null_column in null_columns:
                if isinstance(null_column, str):
                    newData[null_column] = \
                        impCategorical.fit_transform(newData[[null_column]])
                else:
                    newData[null_column] = \
                        impNumeric.fit_transform(newData[[null_column]])
        return newData

    def _createDataset(self, file):
        path = os.path.abspath(file)
        extension = os.path.splitext(path)[1]
        if(extension == ".csv"):
            df = pd.read_csv(path, sep="[,;\t]")
        elif(extension == ".xls" or extension == ".xlsx"):
            df = pd.read_excel(path)
        elif(extension == ".txt"):
            df = pd.read_table(path)
        return df

    def _optimizeModel(self, X, y, model_path, config):
        if not os.path.exists(model_path):
            optimizer = TPOTClassifier(verbosity=2,
                                       config_dict=config)
            optimizer.fit(X, y)
            pipeline = optimizer.fitted_pipeline_
            pickle.dump(optimizer.fitted_pipeline_, open(model_path, 'wb'))
        else:
            pipeline = pickle.load(open(model_path, 'rb'))
        return pipeline

    def showModelStatistics(self):
        # ESTADISTICAS DEL MODELO
        print('----------- FULL PIPELINE METRICS -----------')
        self._printMetrics(self.pipeline)

    def _printMetrics(self, model):
        predictions = model.predict(self.X)
        print()
        print('Confusion Matrix')
        print(confusion_matrix(self.y, predictions))
        print()
        print('Metrics')
        print(f'Accuracy: {accuracy_score(self.y, predictions) * 100} %')
        print(f'F1_score: {f1_score(self.y, predictions) * 100} %')
        print(f'Recall: {recall_score(self.y, predictions) * 100} %')
        print(f'Precision: {precision_score(self.y, predictions) * 100} %')
        print()

    def _separatePipelines(self):
        # SEPARO EL MODELO DEL PIPELINE DE TRANSFORMACION
        # print('--------------TRANSFORMATION PIPELINE---------------')
        trf_steps = []
        for n_step in range(len(self.pipeline.steps)-1):
            # print(f'applying {pipeline.steps[n_step]}...')
            trf_steps.append(self.pipeline.steps[n_step])

        trf_pipeline = Pipeline(steps=trf_steps)
        last_step_name = list(self.pipeline.named_steps.keys())[-1]
        model_used = self.pipeline.named_steps[last_step_name]
        return (trf_pipeline, model_used)

    def showSimplifiedModel(self):
        X_transformed = self._trf_pipeline.fit_transform(self.X)
        # fit simplified model
        # XGB only: output xgb model as text
        self._model_used.get_booster().dump_model('xgbmodel.txt')
        Kmax = 5
        # XGB only
        splitter = DefragModel.parseXGBtrees('./xgbmodel.txt')
        mdl = DefragModel(modeltype='classification')

        mdl.fit(X_transformed, self.y, splitter, Kmax,
                fittype='FAB', featurename=self._feature_names)

        score, cover, coll = mdl.evaluate(X_transformed, self.y)
        print('--------------SIMPLIFIED MODEL----------------')
        print()
        print('Test Error = %f' % (score,))
        print('Test Coverage = %f' % (cover,))
        print('Overlap = %f' % (coll,))
        print()
        print('----- RULES -----')
        print(mdl)
        os.remove('./xgbmodel.txt')

    def predictStudent(self, data, describe=False):
        # pongo algun valor en el target para que nos sea null
        data[self.target] = 'NO'
        newData = self.df.append(data)
        newData, _ = self._preProcessData(newData, self.target)
        student = newData.tail(1)
        y = self.pipeline.predict(student)
        print(f'Preidcted Value: {y}')
        if(describe):
            full_student = self._trf_pipeline.fit_transform(student)
            print(eli5.formatters.as_dataframe.explain_prediction_df(
                                 self._model_used.get_booster(),
                                 full_student[0],
                                 feature_names=self._feature_names))

    def plotSHAPValues(self):
        explainer = shap.TreeExplainer(self._model_used)
        shap_data = self._trf_pipeline.fit_transform(self.X)
        shap_values = explainer.shap_values(shap_data)
        # visualize the first prediction's explanation
        # (use matplotlib=True to avoid Javascript)
        # shap.force_plot(explainer.expected_value, shap_values,
        #                 shap_data, matplotlib=True)
        shap.summary_plot(shap_values, shap_data, plot_type="bar",
                          feature_names=self._feature_names)

    def plotCorrMatrix(self, transformed_features):
        """Plots correlation matrix of numeric attributes."""
        og_features = list(self.X.columns)
        X = pd.DataFrame(columns=self._feature_names,
                         data=self._trf_pipeline.fit_transform(self.X))
        X = X[og_features + transformed_features]
        corr = X.corr()  # calculate correlation among variables
        # creating a null maks
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        # making the plot
        plt.figure()
        sns.heatmap(corr, square=False, mask=mask)
        plt.show()


if __name__ == "__main__":

    reglas_regular = {
        "dataset_path": "Reglas Regular nulls/Base regular.xls",
        "target": "Dos finales por anio",
        "model_path": "Reglas Regular nulls/trainedModel.md",
        "show_feature_importances": True
    }

    reglas_regular_nulls = {
        "dataset_path": "Reglas Regular nulls/Base regular.xls",
        "target": "Dos finales por anio",
        "model_path": "Reglas Regular nulls/trainedModel.md",
        "show_feature_importances": True
    }

    edm_process = AutoEDM()

    edm_process.loadModel()

    edm_process.showModelStatistics()

    # edm_process.showSimplifiedModel()

    # testStudent = pd.DataFrame(columns=['Edad', 'Edad primer anio',
    #                                     'Discapacidad', 'Trabaja',
    #                                     'Tipo Secundario',
    #                                     'Categoria ultimo estudio madre',
    #                                     'Categoria ultimo estudio padre'],
    #                            data=[[25, 18, 'No', 'Si',
    #                                  'BACHILLER', 2, 3]])

    #edm_process.predictStudent(testStudent, describe=True)

    # edm_process.plotSHAPValues()

    # edm_process.plotCorrMatrix([f'Xt{n}' for n in range(15)])
