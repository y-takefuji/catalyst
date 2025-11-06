import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from scipy.stats import spearmanr

# Load the dataset
data = pd.read_csv('HER-catalyst-data-with-DOI.csv')

# Drop the DOI column
data = data.drop('DOI', axis=1)

# Show the shape of the data
print(f"Dataset shape: {data.shape}")

# Separate features and target
X = data.drop('Current_density (mA/cm2)', axis=1)
y = data['Current_density (mA/cm2)']

# Handle 'Composition' column (assuming it should be excluded from modeling as it's string data)
X = X.drop('Composition', axis=1) if 'Composition' in X.columns else X

# Set n as the number of independent features
n = X.shape[1]
print(f"Number of features (n): {n}")

# Function for Random Forest feature importance with 100 trees
def rf_feature_importance(X, y):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    return feature_importance.sort_values('Importance', ascending=False)

# Function for XGBoost feature importance
def xgb_feature_importance(X, y):
    model = xgb.XGBRegressor(random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    return feature_importance.sort_values('Importance', ascending=False)

# Improved Feature Agglomeration function with proper error handling
def get_fa_features(X, n_features=8):
    # Make sure we don't request more features than available
    n_features = min(n_features, X.shape[1])
    
    # Create clusters of features - ensure we don't request more clusters than features
    n_clusters = min(n_features, X.shape[1])
    
    # If we have very few features, just return them all sorted by variance
    if X.shape[1] <= n_features:
        print(f"Only {X.shape[1]} features available, returning all sorted by variance")
        variances = X.var().sort_values(ascending=False)
        return variances.index.tolist()[:n_features]
    
    # Apply Feature Agglomeration
    print(f"Using {n_clusters} clusters for Feature Agglomeration")
    fa = FeatureAgglomeration(n_clusters=n_clusters)
    fa.fit(X)
    clusters = fa.labels_
    
    # Calculate variance for each feature
    variances = X.var().to_dict()
    
    # Create a list of (feature, variance, cluster) tuples
    feature_info = [(col, variances[col], clusters[i]) for i, col in enumerate(X.columns)]
    
    # Sort by variance (descending)
    feature_info.sort(key=lambda x: x[1], reverse=True)
    
    # Track selected clusters to avoid picking multiple features from same cluster
    selected_clusters = set()
    selected_features = []
    
    # Select top features across all clusters
    for feature, variance, cluster in feature_info:
        if len(selected_features) >= n_features:
            break
        
        if cluster not in selected_clusters:
            selected_features.append(feature)
            selected_clusters.add(cluster)
    
    # If we need more features, allow multiple features from same cluster
    if len(selected_features) < n_features:
        remaining = [f for f, v, c in feature_info if f not in selected_features]
        for feature in remaining[:n_features - len(selected_features)]:
            selected_features.append(feature)
    
    print(f"Selected {len(selected_features)} features from {len(selected_clusters)} clusters")
    return selected_features

# Function for Highly Variable Gene Selection (using variance)
def hvgs_importance(X):
    variances = X.var().sort_values(ascending=False)
    return pd.DataFrame({'Feature': variances.index, 'Variance': variances.values})

# Function for Spearman correlation
def spearman_importance(X, y):
    correlations = []
    for column in X.columns:
        corr, _ = spearmanr(X[column], y)
        correlations.append((column, abs(corr)))  # Use absolute correlation
    
    corr_df = pd.DataFrame(correlations, columns=['Feature', 'Correlation'])
    return corr_df.sort_values('Correlation', ascending=False)

# Cross-validation function for regression
def cross_validate_regression(X, y, features, n_folds=5):
    if not features:
        return {'r2_mean': 0, 'r2_std': 0, 'rmse_mean': float('inf'), 'rmse_std': 0}
    
    # Select only the specified features
    X_selected = X[features].copy()
    
    # Initialize model - using RandomForest only
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Initialize cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2_scores = []
    rmse_scores = []
    
    # Perform cross-validation
    for train_idx, test_idx in kf.split(X_selected):
        X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        r2_scores.append(r2)
        rmse_scores.append(rmse)
    
    # Calculate statistics
    return {
        'r2_mean': np.mean(r2_scores),
        'r2_std': np.std(r2_scores),
        'rmse_mean': np.mean(rmse_scores),
        'rmse_std': np.std(rmse_scores)
    }

# Improved stability calculation that accounts for feature order
def calculate_feature_stability(set1, set2, top_feature):
    """
    Calculate stability between feature sets, considering feature order
    
    Parameters:
    set1: List of features from the original dataset
    set2: List of features from the reduced dataset
    top_feature: The top feature that was removed
    
    Returns:
    float: Stability score between 0 and 1
    """
    if not set1 or not set2:
        return 0.0
    
    # Create feature rankings (position dictionaries)
    set1_ranks = {feat: idx for idx, feat in enumerate(set1)}
    set2_ranks = {feat: idx for idx, feat in enumerate(set2)}
    
    # Get remaining features from set1 (exclude the removed top feature)
    remaining_features = [f for f in set1 if f != top_feature]
    
    # If no features remain after removal, return 0
    if not remaining_features:
        return 0.0
    
    # Consider only features that appear in both sets
    common_features = [f for f in remaining_features if f in set2_ranks]
    
    # If no common features, return 0
    if not common_features:
        return 0.0
    
    # Perfect stability means set2 has exactly the same order as set1 (after removing top feature)
    # For each common feature, calculate how many positions it shifted
    total_positions = len(remaining_features)
    sum_position_shifts = 0
    
    for feature in common_features:
        # Get positions in each set (adjusting for the removed top feature in set1)
        pos_set1 = set1_ranks[feature]
        if pos_set1 > set1_ranks.get(top_feature, -1):
            pos_set1 -= 1  # Adjust position if it was after the removed feature
            
        pos_set2 = set2_ranks[feature]
        
        # Calculate position shift
        position_shift = abs(pos_set1 - pos_set2)
        sum_position_shifts += position_shift
    
    # Calculate stability as inverse of normalized position shifts
    max_possible_shifts = total_positions * (total_positions - 1) / 2  # Maximum possible sum of shifts
    if max_possible_shifts == 0:
        return 1.0 if len(common_features) > 0 else 0.0
        
    stability = 1.0 - (sum_position_shifts / max_possible_shifts)
    
    # Adjust for coverage (proportion of features that were preserved)
    coverage = len(common_features) / len(remaining_features)
    
    # Final stability is a combination of order preservation and coverage
    return stability * coverage

# Number of features to select
n_features = min(8, X.shape[1])

# Get features using different methods
rf_ranking = rf_feature_importance(X, y)
xgb_ranking = xgb_feature_importance(X, y)
fa_features = get_fa_features(X, n_features=n_features)
hvgs_ranking = hvgs_importance(X)
spearman_ranking = spearman_importance(X, y)

# Select top n features for each method (set1)
rf_features_set1 = rf_ranking['Feature'].tolist()[:n_features]
xgb_features_set1 = xgb_ranking['Feature'].tolist()[:n_features]
fa_features_set1 = fa_features
hvgs_features_set1 = hvgs_ranking['Feature'].tolist()[:n_features]
spearman_features_set1 = spearman_ranking['Feature'].tolist()[:n_features]

print("\nSelected features for Set 1:")
print(f"RF: {rf_features_set1}")
print(f"XGB: {xgb_features_set1}")
print(f"FA: {fa_features_set1}")
print(f"HVGS: {hvgs_features_set1}")
print(f"Spearman: {spearman_features_set1}")

# Create reduced dataset by removing the highest feature from each method
top_rf_feature = rf_ranking['Feature'].iloc[0] if not rf_ranking.empty else None
X_reduced_rf = X.drop(top_rf_feature, axis=1) if top_rf_feature else X.copy()

top_xgb_feature = xgb_ranking['Feature'].iloc[0] if not xgb_ranking.empty else None
X_reduced_xgb = X.drop(top_xgb_feature, axis=1) if top_xgb_feature else X.copy()

top_fa_feature = fa_features_set1[0] if fa_features_set1 else None
X_reduced_fa = X.drop(top_fa_feature, axis=1) if top_fa_feature else X.copy()

top_hvgs_feature = hvgs_ranking['Feature'].iloc[0] if not hvgs_ranking.empty else None
X_reduced_hvgs = X.drop(top_hvgs_feature, axis=1) if top_hvgs_feature else X.copy()

top_spearman_feature = spearman_ranking['Feature'].iloc[0] if not spearman_ranking.empty else None
X_reduced_spearman = X.drop(top_spearman_feature, axis=1) if top_spearman_feature else X.copy()

# Print removed features
print("\nTop features removed for Set 2:")
print(f"RF: {top_rf_feature}")
print(f"XGB: {top_xgb_feature}")
print(f"FA: {top_fa_feature}")
print(f"HVGS: {top_hvgs_feature}")
print(f"Spearman: {top_spearman_feature}")

# Recompute feature rankings on reduced datasets
rf_ranking_reduced = rf_feature_importance(X_reduced_rf, y)
xgb_ranking_reduced = xgb_feature_importance(X_reduced_xgb, y)
fa_features_reduced = get_fa_features(X_reduced_fa, n_features=min(n_features, X_reduced_fa.shape[1]))
hvgs_ranking_reduced = hvgs_importance(X_reduced_hvgs)
spearman_ranking_reduced = spearman_importance(X_reduced_spearman, y)

# Select top n-1 features from reduced datasets (set2)
n_features_reduced = min(n_features-1, X.shape[1]-1)
rf_features_set2 = rf_ranking_reduced['Feature'].tolist()[:n_features_reduced]
xgb_features_set2 = xgb_ranking_reduced['Feature'].tolist()[:n_features_reduced]
fa_features_set2 = fa_features_reduced
hvgs_features_set2 = hvgs_ranking_reduced['Feature'].tolist()[:n_features_reduced]
spearman_features_set2 = spearman_ranking_reduced['Feature'].tolist()[:n_features_reduced]

print("\nSelected features for Set 2:")
print(f"RF: {rf_features_set2}")
print(f"XGB: {xgb_features_set2}")
print(f"FA: {fa_features_set2}")
print(f"HVGS: {hvgs_features_set2}")
print(f"Spearman: {spearman_features_set2}")

# Calculate stability metrics for each method
rf_stability = calculate_feature_stability(rf_features_set1, rf_features_set2, top_rf_feature)
xgb_stability = calculate_feature_stability(xgb_features_set1, xgb_features_set2, top_xgb_feature)
fa_stability = calculate_feature_stability(fa_features_set1, fa_features_set2, top_fa_feature)
hvgs_stability = calculate_feature_stability(hvgs_features_set1, hvgs_features_set2, top_hvgs_feature)
spearman_stability = calculate_feature_stability(spearman_features_set1, spearman_features_set2, top_spearman_feature)

print("\nFeature Selection Stability (higher is better):")
print(f"Random Forest: {rf_stability:.4f}")
print(f"XGBoost: {xgb_stability:.4f}")
print(f"Feature Agglomeration: {fa_stability:.4f}")
print(f"HVGS: {hvgs_stability:.4f}")
print(f"Spearman: {spearman_stability:.4f}")

# Cross-validate each feature set with Random Forest
print("\n===== Random Forest Cross-Validation Results =====")

# Set 1 Results
print("\nSet 1 Cross-Validation Results (with top feature):")
rf_cv_set1 = cross_validate_regression(X, y, rf_features_set1)
xgb_cv_set1 = cross_validate_regression(X, y, xgb_features_set1)
fa_cv_set1 = cross_validate_regression(X, y, fa_features_set1)
hvgs_cv_set1 = cross_validate_regression(X, y, hvgs_features_set1)
spearman_cv_set1 = cross_validate_regression(X, y, spearman_features_set1)

print(f"Random Forest: R² = {rf_cv_set1['r2_mean']:.4f} ± {rf_cv_set1['r2_std']:.4f}, RMSE = {rf_cv_set1['rmse_mean']:.4f} ± {rf_cv_set1['rmse_std']:.4f}")
print(f"XGBoost: R² = {xgb_cv_set1['r2_mean']:.4f} ± {xgb_cv_set1['r2_std']:.4f}, RMSE = {xgb_cv_set1['rmse_mean']:.4f} ± {xgb_cv_set1['rmse_std']:.4f}")
print(f"Feature Agglomeration: R² = {fa_cv_set1['r2_mean']:.4f} ± {fa_cv_set1['r2_std']:.4f}, RMSE = {fa_cv_set1['rmse_mean']:.4f} ± {fa_cv_set1['rmse_std']:.4f}")
print(f"HVGS: R² = {hvgs_cv_set1['r2_mean']:.4f} ± {hvgs_cv_set1['r2_std']:.4f}, RMSE = {hvgs_cv_set1['rmse_mean']:.4f} ± {hvgs_cv_set1['rmse_std']:.4f}")
print(f"Spearman: R² = {spearman_cv_set1['r2_mean']:.4f} ± {spearman_cv_set1['r2_std']:.4f}, RMSE = {spearman_cv_set1['rmse_mean']:.4f} ± {spearman_cv_set1['rmse_std']:.4f}")

# Set 2 Results
print("\nSet 2 Cross-Validation Results (without top feature):")
rf_cv_set2 = cross_validate_regression(X, y, rf_features_set2)
xgb_cv_set2 = cross_validate_regression(X, y, xgb_features_set2)
fa_cv_set2 = cross_validate_regression(X, y, fa_features_set2)
hvgs_cv_set2 = cross_validate_regression(X, y, hvgs_features_set2)
spearman_cv_set2 = cross_validate_regression(X, y, spearman_features_set2)

print(f"Random Forest: R² = {rf_cv_set2['r2_mean']:.4f} ± {rf_cv_set2['r2_std']:.4f}, RMSE = {rf_cv_set2['rmse_mean']:.4f} ± {rf_cv_set2['rmse_std']:.4f}")
print(f"XGBoost: R² = {xgb_cv_set2['r2_mean']:.4f} ± {xgb_cv_set2['r2_std']:.4f}, RMSE = {xgb_cv_set2['rmse_mean']:.4f} ± {xgb_cv_set2['rmse_std']:.4f}")
print(f"Feature Agglomeration: R² = {fa_cv_set2['r2_mean']:.4f} ± {fa_cv_set2['r2_std']:.4f}, RMSE = {fa_cv_set2['rmse_mean']:.4f} ± {fa_cv_set2['rmse_std']:.4f}")
print(f"HVGS: R² = {hvgs_cv_set2['r2_mean']:.4f} ± {hvgs_cv_set2['r2_std']:.4f}, RMSE = {hvgs_cv_set2['rmse_mean']:.4f} ± {hvgs_cv_set2['rmse_std']:.4f}")
print(f"Spearman: R² = {spearman_cv_set2['r2_mean']:.4f} ± {spearman_cv_set2['r2_std']:.4f}, RMSE = {spearman_cv_set2['rmse_mean']:.4f} ± {spearman_cv_set2['rmse_std']:.4f}")

# Calculate performance changes
print("\nPerformance Change (Set 2 - Set 1):")
print(f"Random Forest: ΔR² = {rf_cv_set2['r2_mean'] - rf_cv_set1['r2_mean']:.4f}, ΔRMSE = {rf_cv_set2['rmse_mean'] - rf_cv_set1['rmse_mean']:.4f}")
print(f"XGBoost: ΔR² = {xgb_cv_set2['r2_mean'] - xgb_cv_set1['r2_mean']:.4f}, ΔRMSE = {xgb_cv_set2['rmse_mean'] - xgb_cv_set1['rmse_mean']:.4f}")
print(f"Feature Agglomeration: ΔR² = {fa_cv_set2['r2_mean'] - fa_cv_set1['r2_mean']:.4f}, ΔRMSE = {fa_cv_set2['rmse_mean'] - fa_cv_set1['rmse_mean']:.4f}")
print(f"HVGS: ΔR² = {hvgs_cv_set2['r2_mean'] - hvgs_cv_set1['r2_mean']:.4f}, ΔRMSE = {hvgs_cv_set2['rmse_mean'] - hvgs_cv_set1['rmse_mean']:.4f}")
print(f"Spearman: ΔR² = {spearman_cv_set2['r2_mean'] - spearman_cv_set1['r2_mean']:.4f}, ΔRMSE = {spearman_cv_set2['rmse_mean'] - spearman_cv_set1['rmse_mean']:.4f}")

# Summary table
print("\n===== Performance Summary Table =====")
print("Method          | Set 1 R² ± std    | Set 2 R² ± std    | ΔR²     | Set 1 RMSE ± std  | Set 2 RMSE ± std  | ΔRMSE    | Stability")
print("-" * 125)
print(f"Random Forest   | {rf_cv_set1['r2_mean']:.4f} ± {rf_cv_set1['r2_std']:.4f} | {rf_cv_set2['r2_mean']:.4f} ± {rf_cv_set2['r2_std']:.4f} | {rf_cv_set2['r2_mean'] - rf_cv_set1['r2_mean']:+.4f} | {rf_cv_set1['rmse_mean']:.4f} ± {rf_cv_set1['rmse_std']:.4f} | {rf_cv_set2['rmse_mean']:.4f} ± {rf_cv_set2['rmse_std']:.4f} | {rf_cv_set2['rmse_mean'] - rf_cv_set1['rmse_mean']:+.4f} | {rf_stability:.4f}")
print(f"XGBoost         | {xgb_cv_set1['r2_mean']:.4f} ± {xgb_cv_set1['r2_std']:.4f} | {xgb_cv_set2['r2_mean']:.4f} ± {xgb_cv_set2['r2_std']:.4f} | {xgb_cv_set2['r2_mean'] - xgb_cv_set1['r2_mean']:+.4f} | {xgb_cv_set1['rmse_mean']:.4f} ± {xgb_cv_set1['rmse_std']:.4f} | {xgb_cv_set2['rmse_mean']:.4f} ± {xgb_cv_set2['rmse_std']:.4f} | {xgb_cv_set2['rmse_mean'] - xgb_cv_set1['rmse_mean']:+.4f} | {xgb_stability:.4f}")
print(f"FA              | {fa_cv_set1['r2_mean']:.4f} ± {fa_cv_set1['r2_std']:.4f} | {fa_cv_set2['r2_mean']:.4f} ± {fa_cv_set2['r2_std']:.4f} | {fa_cv_set2['r2_mean'] - fa_cv_set1['r2_mean']:+.4f} | {fa_cv_set1['rmse_mean']:.4f} ± {fa_cv_set1['rmse_std']:.4f} | {fa_cv_set2['rmse_mean']:.4f} ± {fa_cv_set2['rmse_std']:.4f} | {fa_cv_set2['rmse_mean'] - fa_cv_set1['rmse_mean']:+.4f} | {fa_stability:.4f}")
print(f"HVGS            | {hvgs_cv_set1['r2_mean']:.4f} ± {hvgs_cv_set1['r2_std']:.4f} | {hvgs_cv_set2['r2_mean']:.4f} ± {hvgs_cv_set2['r2_std']:.4f} | {hvgs_cv_set2['r2_mean'] - hvgs_cv_set1['r2_mean']:+.4f} | {hvgs_cv_set1['rmse_mean']:.4f} ± {hvgs_cv_set1['rmse_std']:.4f} | {hvgs_cv_set2['rmse_mean']:.4f} ± {hvgs_cv_set2['rmse_std']:.4f} | {hvgs_cv_set2['rmse_mean'] - hvgs_cv_set1['rmse_mean']:+.4f} | {hvgs_stability:.4f}")
print(f"Spearman        | {spearman_cv_set1['r2_mean']:.4f} ± {spearman_cv_set1['r2_std']:.4f} | {spearman_cv_set2['r2_mean']:.4f} ± {spearman_cv_set2['r2_std']:.4f} | {spearman_cv_set2['r2_mean'] - spearman_cv_set1['r2_mean']:+.4f} | {spearman_cv_set1['rmse_mean']:.4f} ± {spearman_cv_set1['rmse_std']:.4f} | {spearman_cv_set2['rmse_mean']:.4f} ± {spearman_cv_set2['rmse_std']:.4f} | {spearman_cv_set2['rmse_mean'] - spearman_cv_set1['rmse_mean']:+.4f} | {spearman_stability:.4f}")

# Print detailed feature comparison
print("\n===== Detailed Feature Order Comparison =====")
print("Random Forest:")
print(f"  Set 1: {rf_features_set1}")
print(f"  Set 2: {rf_features_set2}")
print("XGBoost:")
print(f"  Set 1: {xgb_features_set1}")
print(f"  Set 2: {xgb_features_set2}")
print("Feature Agglomeration:")
print(f"  Set 1: {fa_features_set1}")
print(f"  Set 2: {fa_features_set2}")
print("HVGS:")
print(f"  Set 1: {hvgs_features_set1}")
print(f"  Set 2: {hvgs_features_set2}")
print("Spearman:")
print(f"  Set 1: {spearman_features_set1}")
print(f"  Set 2: {spearman_features_set2}")
