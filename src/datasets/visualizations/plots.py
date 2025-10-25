from matplotlib import pyplot as plt
import numpy as np
import scipy.stats

def plot_feature_grid(
		df, features, target=None, n_cols=3, figsize=(6.4/2,4.8/2)
):
	n_features = len(features)
	n_rows = int(np.ceil(n_features / n_cols))

	figsize = (n_cols*figsize[0], n_rows*figsize[1])
	fig,axes = plt.subplots(n_rows, n_cols, figsize=figsize)
	axes_flat = axes.ravel()

	for ax,feature in zip(axes_flat, features):
		data = df[feature].dropna()
		color = "slateblue" if feature == target else "steelblue"
		if np.issubdtype(data.dtype, np.integer) and data.nunique() <= 20:
			vals = np.sort(data.unique())
			counts = data.value_counts().reindex(vals, fill_value=0)
			ax.bar(vals.astype(str), counts.values, color=color, edgecolor="black")
		else:
			ax.hist(data, bins=20, color=color, edgecolor="black")
		ax.set_title(str(feature), fontsize=10)
		ax.tick_params(axis="x", labelrotation=45)
	
	for ax in axes_flat[n_features:]:
		ax.axis("off")

	#plt.tight_layout()
	#plt.show()
	return fig, axes

def plot_features_vs_target(
	df, features, target, n_cols=3, figsize=(6.4/2,4.8/2)
):
	assert target not in features, f"target ({target}) can not be in features"
	n_features = len(features)
	n_rows = int(np.ceil(n_features / n_cols))

	figsize = (n_cols*figsize[0], n_rows*figsize[1])
	fig,axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
	axes_flat = axes.ravel()

	for ax_idx, (ax,feature) in enumerate(zip(axes_flat, features)):
		tmp_df = df[[feature,target]].dropna()
		xx = tmp_df[feature]
		yy = tmp_df[target]

		if xx.empty:
			ax.text(0.5, 0.5, "No data", ha="center", va="center")
			ax.set_title(str(feature))
			ax.set_xticks([])
			ax.set_yticks([])
			continue

		is_int = np.issubdtype(xx.dtype, np.integer)
		n_unique = xx.nunique()
		discrete = is_int and n_unique <= 20

		if discrete:
			values = np.sort(xx.unique())
			values_to_pos = {value:ii for ii, value in enumerate(values)}
			x_pos = xx.map(values_to_pos).astype(float).values
			ax.scatter(x_pos, yy.values, s=20, color="cyan", alpha=0.25, edgecolors="black", linewidths=0.2)
			ax.set_xticks(range(len(values)))
			ax.set_xticklabels(list(map(str,values)), rotation=45)
			ax.set_xlabel(str(feature))
		else:
			ax.scatter(xx.values, yy.values, s=20, color="cyan", alpha=0.25, edgecolors="black", linewidths=0.2)
			ax.set_xlabel(str(feature))

		ax.set_ylabel(str(target) if ax_idx % n_cols == 0 else "")
		ax.set_title(str(feature))

	for ax in axes_flat[len(features):]:
		ax.axis("off")

	#plt.tight_layout()
	#plt.show()
	return fig, axes

def plot_feature_pairs(df, features, figsize=(1.75, 1.5)):
	n_features = len(features)
	n_rows = n_features
	n_cols = n_features

	figsize = (n_cols*figsize[0], n_rows*figsize[1])
	fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

	for ii,feature_i in enumerate(features):
		for jj,feature_j in enumerate(features):
			ax = axes[ii,jj]

			if ii == jj:
				ax.hist(df[feature_i], bins=20, color="slateblue", alpha=1)
			else:
				ax.scatter(df[feature_j], df[feature_i], s=5, alpha=0.25, color="steelblue")

			# Label only outer plots for readability
			if ii == n_features - 1:
				ax.set_xlabel(feature_j)
			else:
				ax.set_xticklabels([])
			if jj == 0:
				ax.set_ylabel(feature_i)
			else:
				ax.set_yticklabels([])

	#plt.tight_layout()
	#plt.show()
	return fig, axes

def compare_feature_distribution(
	train_df, test_df, feature, bins=50, density=True, ax=None, figsize=(6.4,4.8)
):
	train_values = train_df[feature].dropna()
	test_values = test_df[feature].dropna()

	ks_stat, p_value = scipy.stats.ks_2samp(train_values, test_values)

	min_val = min(train_values.min(), test_values.min())
	max_val = max(train_values.max(), test_values.max())
	bin_edges = np.linspace(min_val, max_val, bins + 1)

	ax_is_none = ax is None
	if ax_is_none:
		fig, ax = plt.subplots(figsize=figsize)

	ax.hist(train_values, bins=bin_edges, density=density, alpha=0.5, color="steelblue", label="Train")
	ax.hist(test_values, bins=bin_edges, density=density, alpha=0.5, color="slateblue", label="Test")

	ax.set_title(f"{feature}\nKS={ks_stat:.3f}, p={p_value:.1e}")
	ax.set_xlabel(feature)
	ax.set_ylabel("Density" if density else "Count")

	if ax_is_none:
		fig.tight_layout()
		plt.show()

	return ks_stat, p_value

def compare_feature_distributions_grid(
	train_df, test_df, features, n_cols=3, bins=50, density=True, figsize=(6.4/2,4.8/2)
):
	n_features = len(features)
	n_rows = int(np.ceil(n_features / n_cols))

	figsize = (n_cols*figsize[0], n_rows*figsize[1])
	fig,axes = plt.subplots(n_rows, n_cols, figsize=figsize)
	axes_flat = axes.ravel()

	results = {}
	for ax,feature in zip(axes_flat, features):
		ks_stat, p_value = compare_feature_distribution(
			train_df, test_df, feature, bins=bins, density=density, ax=ax
		)
		results[feature] = {"ks_stat": ks_stat, "p_value": p_value}

	for ax in axes_flat[n_features:]:
		ax.axis("off")

	handles, labels = axes_flat[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))

	return fig, axes, results

