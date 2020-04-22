import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator
import pickle

class MLArchive:
	"""MLArchive class to hold the training models history.

	This class should be used to record our models and 
	results from the iterative training process.
	
	"""

	schema = [
		'id', 'technique', 'model', 'metric', 'date', 'train_res',
		'devel_res', 'test_res', 'params', 'train_samples',
		'test_samples', 'train_hist', 'columns', 'packages'
	]

	def __init__(self, filename: str = None, models_path: str = 'model_archive/'):
		if filename == None:
			self.__model_path = models_path
			self.__ranked_models = pd.DataFrame(columns=self.schema)
		else:
			self.load_archive(filename)

	def save_model(self, mod: object, metric: str, train_res: float,
			test_res: float, devel_res: float = None,
			train_samples: int = None, test_samples: int = None, 
			train_hist: dict = None, columns: list = None,
			packages: str = None) -> None:

		"""
		This method saves a new entry for a new model and rank it 
		based on the test result metric.
		
		Parameters
		----------
		mod : object
			The trained model.
		metric : str
			The metric which was used.
		train_res: float,
			Result of the chosen metric in the train set.
		test_res: float
			Result of the chosen metric in the test set.
		devel_res: float, default = None,
			Numer of train samples.
		train_samples: int, default = None
			Number of train samples.
		test_samples: int, default = None
			Number of test samples.
		train_hist: dict, default = None
			The training history per training step.
		columns: list, default = None
			The model input features.
		packages: str, default = None
			To preserve reciprocity could contain the model requirements.
		"""

		model = {}
		model['model'] = mod
		model['metric'] = metric
		model['train_res'] = train_res
		model['devel_res'] = devel_res
		model['test_res'] = test_res
		model['train_samples'] = train_samples
		tech = ''
		if isinstance(mod, BaseEstimator):
			params = mod.get_params()
			name = type(mod).__name__
			if name == 'Pipeline':
				name = type(mod[-1]).__name__
			tech = name

		# TODO: Keras NN, XGBoost, LightGBM, ...
		elif isinstance(mod, None):
			tech = 'TODO'
			pass # TODO
		elif isinstance(mod, None):
			tech = 'TODO'
			pass # TODO 
		elif isinstance(mod, None):
			tech = 'TODO'
			pass # TODO 
		else:
			print("Sorry, we haven't implemented save for "\
				  "this kind of model. Please implement it on "\
				  "save_model and submit a pull request. Thanks!")
		
		now = datetime.now()
		model['technique'] = tech
		model['date'] = now.strftime("%d/%m/%Y %H:%M:%S")
		model['id'] = now.strftime(tech+"%d%m%Y%H%M%S")

		df = self.__ranked_models

		df = df.append(model, ignore_index=True)

		df = df.sort_values(by='test_res', ascending=False)
		df = df.reset_index(drop=True)
		pos = str(df.loc[df['id']==model['id']].index[0])
		self.__ranked_models = df
		print('Model '+model['id']+' added in position: '+pos)

	def load_model(self, id: str) -> object:
		""" 
		This method load a previously trained model from the
		archive using it's id.

		Parameters
		----------
		id: str
			id of the model to load.
		"""
		# TODO
		pass

	def load_best_model(self) -> object:
		"""
		Load a previously trained model with the best result
		in the test set.
		"""
		return load_model(self.__ranked_models.iloc[0, 'id'])

	def ranked_models(self, lim: int = None, cols = range(8)) -> pd.DataFrame:
		"""
		Retrieve the registry of trained models.
		
		Parameters
		----------
		lim: int
			Number of registries to load.
		cols: Index or array-like
			Column labels to use for resulting frame.
		"""
		return self.__ranked_models.iloc[:lim, cols]

	def get_path(self) -> str:
		"""
		Get the path where we write the models.
		"""
		return self.__model_path

	def save_archive(self, filename: str) -> None:
		"""
		Write the archive to file.
		
		Parameters
		----------
		filename: str
			File to write on.
		"""
		pickle.dump(self, open(filename, "wb"))

	def load_archive(self, filename: str) -> None:
		"""
		Load the archive from file.

		Parameters
		----------
		filename: str
			File to load from.
		"""
		data = pickle.load(open(filename, "rb"))
		self.__model_path = data.get_path()
		self.__ranked_models = \
			self.ranked_models(cols = range(len(self.schema)))


from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate


if __name__ == "__main__":
	arch = MLArchive()
	datafile = "data_03-31-2020.h5"
	data = pd.read_hdf(datafile, key='df')
	y = data.p1_win
	X = data.iloc[:, :-1]

	names = ["Neural Net", "Log reg", "Random Forest", "Decision Tree"]

	classifiers = [
		Pipeline([('scaler', RobustScaler()),
				  ('mlp', MLPClassifier(alpha=0.5,
										hidden_layer_sizes=(64,4),
										activation='relu',
										max_iter=100000,
										validation_fraction=0.3,
										early_stopping=True, 
										random_state=42))]),
		LogisticRegression(C=0.0001,max_iter=10000, 
					 intercept_scaling=False,
					 fit_intercept=False),
		RandomForestClassifier(max_depth=8, n_estimators=50, 
							   max_samples=0.7, random_state=42),
		DecisionTreeClassifier(max_depth=16),]
	
	# iterate over classifiers
	print('------------------------------------------------')
	for name, clf in zip(names, classifiers):
		cv_results = cross_validate(clf, X, y, cv=5,
									return_train_score=True)
		print(name + ' train: ' + ('%.2f' % cv_results['train_score'].mean()).lstrip('0'))
		print(name + ' test: ' + ('%.2f' % cv_results['test_score'].mean()).lstrip('0'))

		arch.save_model(clf, metric='accuracy', 
			train_res=cv_results['train_score'].mean(), 
			test_res=cv_results['test_score'].mean())
		print('------------------------------------------------')
	arch.save_archive('MLModel.arch')
	
	arch2 = MLArchive('MLModel.arch')
	print('done')