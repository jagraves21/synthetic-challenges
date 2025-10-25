import numpy as np

def mae(y_true, y_pred):
	return np.mean(np.abs(y_pred - y_true))

def mse(y_true, y_pred):
	return np.mean( np.square(y_pred - y_true) )

def rmse(y_true, y_pred):
	return np.sqrt(mse(y_true, y_pred))

def mape(y_true, y_pred, eps=1e-8):
	y_true = np.clip(y_true, eps, None)
	return 100 * np.mean(np.abs((y_true - y_pred) / y_true))

def smape(y_true, y_pred, eps=1e-8):
	denominator = np.abs(y_true) + np.abs(y_pred) + eps
	return 200 * np.mean(np.abs(y_pred - y_true) / denominator)

def r2_score(y_true, y_pred, eps=1e-8):
	ss_res = np.sum( np.square(y_true - y_pred) )
	ss_tot = np.sum( np.square((y_true - np.mean(y_true)) ))
	return 1 - ss_res / (ss_tot + eps)

def evaluate_all_metrics(y_true, y_pred):
	return {
		"MAE": mae(y_true, y_pred),
		"MSE": mse(y_true, y_pred),
		"RMSE": rmse(y_true, y_pred),
		"MAPE": mape(y_true, y_pred),
		"sMAPE": smape(y_true, y_pred),
		"R2": r2_score(y_true, y_pred)
	}

