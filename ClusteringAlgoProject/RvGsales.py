import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import matplotlib.cm as cm
Test_K = None
plt.rcParams["figure.figsize"] = (7,5)
plt.rcParams["axes.grid"] = True

## Load CSV file and clean data

csv_path = Path(__file__).parent / "vgsales.csv"
df = pd.read_csv(csv_path)

df.columns = [c.strip().replace(" ", "_") for c in df.columns]

req = {"Name", "Platform", "Year", "Genre", "Publisher", "NA_Sales", "EU_Sales", "Other_Sales"}
missing = req - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in data: {missing}")

#Set important categories and make year a numeric category

df = df.dropna(subset=["Genre", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"])
df["Year"] = pd.to_numeric(df["Year"], errors = "coerce")
df["Year"] = df["Year"].fillna(df["Year"].median())

#
sales_cols = ["NA_Sales", "EU_Sales", "JP_Sales","Other_Sales"]
df["Tot_Sales"] = df[sales_cols].sum(axis=1)
df = df[df["Tot_Sales"] > 0]

#Determine which region had the most sales for a game
df["Region_Winner"] = df[sales_cols].idxmax(axis=1).str.replace("_Sales", "")

#find preference per game
for c in sales_cols:
    df[c+"_Share"] = df[c] / df["Tot_Sales"]

# Build Clusters for genres
g = df.groupby("Genre")

genre_profile = pd.DataFrame({
    "n_games": g.size(),
    "year_mean": g["Year"].mean(),
    "tot_na": g["NA_Sales"].sum(),
    "tot_eu": g["EU_Sales"].sum(),
    "tot_jp": g["JP_Sales"].sum(),
    "tot_oth": g["Other_Sales"].sum(),
    "tot_all": g["Tot_Sales"].sum()})

#region share means
for c in sales_cols:
    genre_profile[c.replace("_Sales", "_share_mean")] = g[c+"_Share"].mean()

#Create Winning rate for fraction of games for each region
winner_rates = (g["Region_Winner"].value_counts(normalize=True).unstack().reindex(columns=["NA","EU","JP","Other"], fill_value=0))

winner_rates.columns = [f"Winner_rate_{c.lower()}" for c in winner_rates.columns]

#Bring the rates together
genre_profile = genre_profile.join(winner_rates).fillna(0)

for c in ["tot_na", "tot_eu", "tot_jp", "tot_oth"]:genre_profile[c+"_log1p"] = np.log1p(genre_profile[c])

def entropy(p):
    p = np.asarray(p)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

genre_profile["winner_entropy"] = winner_rates.apply(lambda row: entropy(row.values), axis = 1)

#Final feature matrix for clustering
feature_cols = ["NA_share_mean", "EU_share_mean", "JP_share_mean", "Other_share_mean", "Winner_rate_na", "Winner_rate_eu", "Winner_rate_jp", "Winner_rate_other",
                "year_mean","tot_na_log1p", "tot_eu_log1p", "tot_jp_log1p", "tot_oth_log1p","winner_entropy"]

X_raw = genre_profile[feature_cols].copy()

#Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

#Create PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
print("PCA explained variance in 2D:", round(pca.explained_variance_ratio_.sum(), 3))

#K-Means
def findk_silhouette(X, min_k=2, max_k=8):
    max_k = min(max_k, max(2, len(X)-1))
    best_k = None
    best_score = -1
    best_km = None
    for k in range(min_k, max_k+1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        if len(set(labels)) == 1:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k = k
            best_score = score 
            best_km = km
    return best_k, best_score, best_km

best_k, sil, km = findk_silhouette(X, 2, 8)

if km is None:
    best_k, sil, km = 3, float("nan"), KMeans(n_clusters = 3, n_init=10, random_state=42).fit(X)
    
if Test_K is not None:
    best_k = int(Test_K)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
    
def plot_sil(X, labels, title):
    k = len(np.unique(labels))
    if k < 2:
        print(f"Silhouette needs >= 2 clusters; got {k}. Skipping: {title}")
        return

    # Overall score/values per sample
    s_avg = silhouette_score(X, labels)
    s_vals = silhouette_samples(X, labels)

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.set_title(f"{title}\nAverage silhouette = {s_avg:.3f}")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster")

    ax1.set_xlim([-0.1, 1.0])
    ax1.set_ylim([0, len(X) + (k + 1) * 10])

    y_lower = 10
    for i in range(k):
        vals_i = s_vals[labels == i]
        vals_i.sort()
        size_i = vals_i.shape[0]
        y_upper = y_lower + size_i

        color = cm.nipy_spectral(float(i) / k)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, vals_i,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_i, str(i))
        y_lower = y_upper + 10  # 10 for spacing

    ax1.axvline(x=s_avg, color="red", linestyle="--", linewidth=1.5)
    ax1.set_yticks([])  # labels on the bars already
    ax1.set_xticks(np.linspace(-0.1, 1.0, 7))
    plt.tight_layout()
    plt.show()
    

genre_profile["KMeans_Cluster"] = km.predict(X)
print(f"Best K (KMeans): {best_k},silhouette: {sil:.3f}" if sil==sil else f"KMeans used k={best_k}")

#Hierarchical Clustering
agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
genre_profile["Agglo_Cluster"] = agg.fit_predict(X)

#GMM
try: 
    gmm = GaussianMixture(n_components=best_k, covariance_type="full", random_state=42)
    genre_profile["GMM_Cluster"] = gmm.fit_predict(X)
    #Silhouette plots
    plot_sil(X, genre_profile["KMeans_Cluster"].values, f"Silhouette Plot (KMeans, k={best_k})")

             
    if "GMM_Cluster" in genre_profile.columns:
        plot_sil(X, genre_profile["GMM_Cluster"].values, f"Silhouette Plot (GMM, k={best_k})")

    for i in range(best_k):
        genre_profile[f"GMM_p{i}"] = gmm.predict_proba(X)[:, i]
except Exception as e:
    print("error: ", e)
#Cluster profile assist
def prof_clusters(df_gp, cluster_col):
    grp = df_gp.groupby(cluster_col)
    
    #what goes inside each cluster
    winner_cols = ["Winner_rate_na", "Winner_rate_eu","Winner_rate_jp", "Winner_rate_other"]
    share_cols = ["NA_share_mean", "EU_share_mean", "JP_share_mean", "Other_share_mean"]
    
    summary = pd.DataFrame({"n_genres": grp.size(),"avg_winner_rate_na": grp[winner_cols[0]].mean(),
                            "avg_winner_rate_eu": grp[winner_cols[1]].mean(),
                            "avg_winner_rate_jp": grp[winner_cols[2]].mean(),
                            "avg_winner_rate_oth": grp[winner_cols[3]].mean(),
                            "avg_share_rate_na": grp[share_cols[0]].mean(),
                            "avg_share_rate_eu": grp[share_cols[1]].mean(),
                            "avg_share_rate_jp": grp[share_cols[2]].mean(),
                            "avg_share_rate_oth": grp[share_cols[3]].mean(),
                            "avg_year_mean": grp["year_mean"].mean(),
                            "avg_winner_entropy": grp["winner_entropy"].mean(),
                            "sample_genres": grp.apply(lambda g: ", ".join(g.index.tolist()[:6]))})
    #find what region looks dominant in a cluster
    summary["dominant_region_by_winner_rate"] = summary[["avg_winner_rate_na","avg_winner_rate_eu","avg_winner_rate_jp", "avg_winner_rate_oth"]].idxmax(axis=1).str.replace("avg_winner_rate_","").str.upper()
    return summary

print("/n KMeans Cluster profile (genre-level)")
print(prof_clusters(genre_profile, "KMeans_Cluster"))


#Visualization
def scatter(xy, labels, title):
    plt.figure()
    plt.scatter(xy[:,0], xy[:,1], s=60, c = labels)
    for (gx,gy), text in zip(xy, genre_profile.index):
        plt.text(gx, gy, text, fontsize=8)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()
        
scatter(X_pca, genre_profile["KMeans_Cluster"], f"PCA of Genres (KMeans, k = {best_k})")

#Dendrogram
Z = linkage(X, method="ward")
plt.figure(figsize=(12, 6.5))
dendrogram(Z, labels=genre_profile.index.tolist(), leaf_rotation=0)
plt.title("Genre Dendrogram (Ward Linkage)")
plt.tight_layout()
plt.show()

#Find and Show Dominant Genre in each Region
d_genre = (df.groupby("Genre")[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].sum())

dom_tot = d_genre.idxmax(axis=0)
print("\n Dominate Genre per Region:")
print(dom_tot)
#Save Outputs
out_path = Path("genre_clusters.csv")
cols_to_save = feature_cols + ["KMeans_Cluster","Agglo_Cluster"] + ([c for c in genre_profile.columns if c.startswith("GMM_")] if "GMM_Cluster" in genre_profile.columns else[])

genre_profile[cols_to_save].to_csv(out_path)
print(f"/nSaved genre cluster features & assignments to {out_path.resolve()}")
    

    
