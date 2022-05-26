from collections import Counter
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import get_cmap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from django.contrib import messages
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.cross_decomposition import CCA

def index(request):

    context = {}
    global attribute,n_components,metric,perplexity,target


    n_components=2
    metric='euclidean'
    perplexity=20
    target=''



    if request.method == 'POST':

        uploaded_file = request.FILES['document']
        attribute = request.POST.get('DRA')
        n=request.POST.get('n-components')
        m=request.POST.get('Distance-Metric')
        p=request.POST.get('Perplexity')
        t=request.POST.get('Target')


        #setting components if user has entered those
        if n:
            n_components=int(n)
        if m:
            metric=m
        if p:
            perplexity=int(p)
        if t:
            target=t


        #check if this file ends with csv
        if uploaded_file.name.endswith('.csv'):
            savefile = FileSystemStorage()
            name = savefile.save(uploaded_file.name, uploaded_file) #gets the name of the file
            print(name)
            d = os.getcwd()
            file_directory = d+'\media\\'+name #saving the file in the media directory
            print(file_directory)
            data_target = pd.read_csv(file_directory)
            if not target:
                print(target)
                print(data_target.columns[0])
                target = data_target.columns[0]
                print(target)
            readfile(file_directory)

            #calling DR algorithms as per user choice
            if attribute=='PCA':
                pca_reduction()
            elif attribute=='SVD':
                svd_reduction()
            elif attribute == 't-SNE':
                t_sne_reduction()
            elif attribute=='UMAP':
                umap_reduction()
            elif attribute=='CCA':
                cca_reduction()
            return redirect(results)

        else:
            messages.warning(request, 'File was not uploaded. Please use .csv file extension!')


    return  render(request, 'index.html', context)

def readfile(filename):
    colors = ['C{}'.format(i) for i in range(10)]
    global data,my_file,df,cmap
    my_file = pd.read_csv(filename)
    data = pd.DataFrame(data=my_file, index=None)
    encoder = LabelEncoder()

    for col in data:
        data[col] = encoder.fit_transform(data[col])

    data = data.dropna()
    df=pd.DataFrame()

   
    for i in data:
        if i != target:
            df[i]=data[i]
    num_classes = len(np.unique(data[target]))
    cmap = LinearSegmentedColormap.from_list('', colors[:num_classes])


def  pca_reduction():
    features = []
    df = data
    for x in df:
        features.append(x)

    pca = PCA()
    components = pca.fit_transform(df.values)
    #print(components)
    labels = {
        str(i): f"PC {i + 1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }
    plt.scatter(components[:, 0], components[:, 1], c=data[target], cmap=cmap)
    plt.show()
    '''fig = px.scatter_matrix(components,labels=labels,color=data[target])
    fig.update_traces(diagonal_visible=False)
    fig.show()'''

def t_sne_reduction():
    df_subset=pd.DataFrame()
    tsne = TSNE(n_components=n_components, perplexity=perplexity,metric=metric)
    tsne_results = tsne.fit_transform(data)
    print(tsne_results)
    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(dpi=100)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=data[target], cmap=cmap, s=5)
    plt.show()

def svd_reduction():
    u, s, v = np.linalg.svd(data, full_matrices=True)
    var_explained = np.round(s ** 2 / np.sum(s ** 2), decimals=3)
    labels = ['SV' + str(i) for i in range(1, 3)]
    svd_df = pd.DataFrame(u[:, 0:2], index=data[target].tolist(), columns=labels)
    svd_df = svd_df.reset_index()
    svd_df.rename(columns={'index': target}, inplace=True)
    sns.scatterplot(x="SV1", y="SV2", hue=data[target],
                    palette=cmap,
                    data=svd_df, s=100,
                    alpha=0.7)
    plt.xlabel('SV 1: {0}%'.format(var_explained[0] * 100), fontsize=16)
    plt.ylabel('SV 2: {0}%'.format(var_explained[1] * 100), fontsize=16)
    plt.show()


def cca_reduction():
    X=df
    y=data[[target]]
    cca = CCA(n_components=n_components)
    my_cca=cca.fit(df,y)
    # Obtain the rotation matrices
    xrot = my_cca.x_rotations_
    yrot = my_cca.y_rotations_

    # Put them together in a numpy matrix
    xyrot = np.vstack((xrot, yrot))

    nvariables = xyrot.shape[0]

    plt.figure(figsize=(10, 10))
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))

    # Plot an arrow and a text label for each variable
    for var_i in range(nvariables):
        x = xyrot[var_i, 0]
        y = xyrot[var_i, 0]

        plt.arrow(0, 0, x, y)
        plt.text(x, y, data.columns[var_i], color='red' )

    plt.show()




def umap_reduction():
    df_subset = pd.DataFrame()
    reducer=umap.UMAP(n_components=n_components, metric=metric,n_neighbors=perplexity)
    umap_results = reducer.fit_transform(data)

    df_subset['umap-2d-one'] =umap_results[:, 0]
    df_subset['umap-2d-two'] =umap_results[:, 1]
    plt.figure(figsize=(10, 10))
    plt.scatter(umap_results[:, 0], umap_results[:, 1], c=data[target], cmap=cmap, s=5)
    plt.show()


def results(request):
    return render(request, 'results.html')