###############################################################################

from sklearn.model_selection import StratifiedKFold

from sklearn.svm import LinearSVC as SVM
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GB


class ConfigHelper():
	k_folds = 10

	@staticmethod
	def k_fold_cv(labels):

		skf = StratifiedKFold(n_splits=ConfigHelper.k_folds, shuffle=True)
		return skf.split(X=range(len(labels)), y=labels)

	@staticmethod
	def get_models():

		return [SVM(dual=False, tol=0.0001, C=1, max_iter=1000),
				MLP(hidden_layer_sizes=(100, ), alpha=0.0001,
					activation="relu", learning_rate_init=0.001,
					tol=0.0001, max_iter=200),
				RF(n_estimators=10, max_depth=None, min_samples_split=2,
				   bootstrap=True, n_jobs=-1),
				GB(n_estimators=100, learning_rate=0.1, subsample=1.0,
				   max_depth=3, min_samples_split=2, tol=0.0001)
				]