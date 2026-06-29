"""Build A_geo for alternative KNN k (Tier-1 sensitivity), reusing the exact
centroids and Step-4/5 logic of build_a_geo.py — WITHOUT re-running geopandas.

The row-normalized A_geo is independent of the Earth radius (it cancels in the
1/D ratio), and KNN selection is monotone in D, so this reproduces a_geo.npy
exactly for k=5 (verified below) and gives accurate k=3 / k=8 graphs.

Run from the repo root (needs data/a_geo_centroids.csv + data/a_geo.npy):
  python build_a_geo_k.py
Outputs: a_geo_k3.npy, a_geo_k8.npy
"""
import numpy as np
import pandas as pd

R = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlam = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = (np.sin(dphi / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2)
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


df = pd.read_csv('data/a_geo_centroids.csv')
lats, lons = df['lat'].values, df['lon'].values
N = len(df)
D = haversine_km(lats[:, None], lons[:, None], lats[None, :], lons[None, :])
np.fill_diagonal(D, np.inf)


def build(K):
    knn_idx = np.argpartition(D, K, axis=1)[:, :K]
    A = np.zeros((N, N))
    for i in range(N):
        for j in knn_idx[i]:
            A[i, j] = 1.0 / D[i, j]
    A = A / A.sum(axis=1, keepdims=True)
    assert np.allclose(A.sum(1), 1.0)
    assert (A > 0).sum(1).min() == K
    return A


# --- correctness: k=5 must reproduce the production a_geo.npy ---
A5 = build(5)
ref = np.load('data/a_geo.npy')
ok = np.allclose(A5, ref)
print(f'k=5 reproduces a_geo.npy: {ok}  (max|diff|={np.abs(A5 - ref).max():.2e})')
assert ok, 'k=5 mismatch -> centroid order or logic differs; abort'

for K in (3, 8):
    A = build(K)
    np.save(f'data/a_geo_k{K}.npy', A)
    dens = (A > 0).sum() / (N * (N - 1)) * 100
    print(f'saved data/a_geo_k{K}.npy  density={dens:.2f}%  '
          f'(k/(N-1)={K/(N-1)*100:.2f}%)')
print('DONE')
