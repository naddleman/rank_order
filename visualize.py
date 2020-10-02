"""Visualizations for clustered rank-order data"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA

pca = PCA(n_components=2)
kpca = KernelPCA(n_components = 2, kernel="rbf", fit_inverse_transform=True, gamma=10)

def normalized_rankings(df, columns=['Lust', 'Envy', 'Greed', 'Sloth', 'Wrath',
                                        'Pride', 'Gluttony']):
    """Normalize (in place for now) rankings to mean=0, variance=1 for PCA"""
    df[columns] = (df[columns] - 4) / 2 # hard coding n=7 case for now


def add_pca_to_normalized_df(df, columns=['Lust', 'Envy', 'Greed', 'Sloth', 
                                        'Wrath', 'Pride', 'Gluttony']):
    relevant = df[columns]
    principal_components = pca.fit_transform(relevant)
    pca_df = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])
    out = pd.concat([df, pca_df], axis=1)
    return out

def add_kpca_to_normalized_df(df, columns=['Lust', 'Envy', 'Greed', 'Sloth', 
                                        'Wrath', 'Pride', 'Gluttony']):
    relevant = df[columns]
    kprincipal_components = kpca.fit_transform(relevant)
    kpca_df = pd.DataFrame(data = kprincipal_components, columns = ['PC1', 'PC2'])
    out = pd.concat([df, kpca_df], axis=1)
    return out

df_fn = pd.read_csv('data_frame.csv', index_col=0)
ds2 = np.load('distances.npy')
kpca2 = KernelPCA(n_components = 2, kernel = "precomputed")
ker2 = np.exp(-np.square(ds2 * .002))
kpcs = kpca2.fit_transform(ker2)
kpcs_df = pd.DataFrame(data=kpcs, columns = ['PC1', 'PC2'])
df_custom_kernel = pd.concat([df_fn, kpcs_df], axis=1)
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal component 1')
ax.set_ylabel('Principal component 2')
ax.set_title('Two principal components for sins data (Kendall Tau Kernel)')
for cluster in [0,1]:
    indices = df_custom_kernel['cluster'] == cluster
    ax.scatter(df_custom_kernel.loc[indices, 'PC1'],
               df_custom_kernel.loc[indices, 'PC2'],
               c = ['r', 'b'][cluster])
plt.show()

