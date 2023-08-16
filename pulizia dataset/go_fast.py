import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import numpy as np
import operator

df = pd.read_csv(r"./dataset.csv")

df2 = df[["track_id", "writers", "track_name", "track_pos"]]
df2_dict = df2.to_dict(orient='records')

d = []

for f in df2_dict:
    d.extend(f["writers"].split("-"))

set_d = set([a.strip() for a in d])
dict_vett_writers = []
new_dict_vect_writers = []
support = []
support2 = ""

for writer in set_d:
    for d in df2_dict:
        if writer in [a.strip() for a in d["writers"].split("-")]:
            support.append({"name": d["track_name"], "pos": d["track_pos"], "year": int(
                d["track_id"].split("_")[0])})
            support2 += d["track_name"]+" - " + \
                str(d["track_pos"]) + " - " + \
                str(d["track_id"].split("_")[0]) + ","
    dict_vett_writers.append({"writer": writer, "tracks": support})
    new_dict_vect_writers.append(
        {"writer": writer, "tracks": support2[:-1].split(",")})
    support = []
list_edges = []
l_e_new = []
list_weights = []

for dic in dict_vett_writers:
    for dic2 in dict_vett_writers:
        for song in dic["tracks"]:
            if song in dic2["tracks"] and dic["writer"] != dic2["writer"]:
                if {"source": dic["writer"], "target": dic2["writer"]} in list_edges:
                    index = list_edges.index(
                        {"source": dic["writer"], "target": dic2["writer"]})
                    list_weights[index] += (10-song["pos"])
                else:
                    common_tracks = []
                    list_edges.append(
                        {"source": dic["writer"], "target": dic2["writer"]})
                    for t1 in dic["tracks"]:
                        if t1 in dic2["tracks"]:
                            common_tracks.append(t1)
                    l_e_new.append(
                        {"source": dic["writer"], "target": dic2["writer"], "tracks": common_tracks, "year": t1["year"]})
                    list_weights.append(1 + (10-song["pos"]))

for weight in list_edges:
    a = weight["source"]
    b = weight["target"]
    index = list_edges.index(weight)
    try:
        list_edges.remove({"source": b, "target": a})
        list_weights.remove(list_weights[index])
    except:
        pass

lista = list(set([str(a["writer"]).strip() for a in dict_vett_writers]))
# print(True if ("M. Vicino") in lista else False)
'''for w in lista:
    if w not in [b["source"] for b in list_edges]:
        lista.remove(w)'''

# top songwriters per numero di canzoni
v_singers = ['D. Pace', 'F. Migliacci', 'M. Panzeri', 'Mogol', 'A. Cogliati']
tabella = []
for singer in v_singers:
    val_singer = 0
    val_placed = 0
    for i, row in df2.iterrows():
        if singer in [a.strip() for a in row["writers"].split("-")] and row["track_pos"] <= 1:
            val_singer += 1
        if singer in [a.strip() for a in row["writers"].split("-")]:
            val_placed += 1
    tabella.append({"writer": singer, "val_top3": val_singer,
                   "val_placed": val_placed})
    # print("fine "+singer)

G = nx.Graph()
G.add_nodes_from(lista)
ebunch = [(list_edges[i]["source"], list_edges[i]["target"], {
           "weight": list_weights[i]}) for i in range(len(list_edges))]
G.add_edges_from(ebunch)

'''gMogol = nx.Graph()
gMogol.add_nodes_from(lista)
ebunchMogol = [(list_edges[i]["source"],list_edges[i]["target"],{"weight":list_weights[i]}) for i in range( len(list_edges)) if list_edges[i]["source"] == "Mogol" or list_edges[i]["target"] == "Mogol"]
gMogol.add_edges_from(ebunchMogol)
gMogol.remove_nodes_from(list(nx.isolates(gMogol)))'''

# degree centrality
print("degree centrality")
deg_centrality = nx.degree_centrality(G)
# nx.write_gexf(deg_centrality, "deg_cent.gexf")
print(deg_centrality, '\n')

'''import collections
from scipy.interpolate import make_interp_spline
#degree distribution
print("degree distribution")
print(nx.degree_histogram(G),'\n') #find probability
#plot degree distribution as a bar plot with a spline on top of the same data points
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
x = range(len(nx.degree_histogram(G)))
y = nx.degree_histogram(G)
xnew = np.linspace(min(x), max(x), 300)
spl = make_interp_spline(x, y, k=3)  # type: BSpline
power_smooth = spl(xnew)
plt.bar(deg, cnt, width=0.80, color="b")
plt.plot(xnew, power_smooth, color="red")   
plt.title("Degree Distribution Graph")
plt.ylabel("Count")
plt.xlabel("Degree")
plt.show()'''


'''
plt.plot(xnew, power_smooth, color="red")
plt.show()'''


# pagerank
print("pagerank")
prank = nx.pagerank(G, weight="weight")
print(prank, '\n')

# average clustering coefficient
print("average clustering coefficient")
print(nx.average_clustering(G, weight="weight"), '\n')


# k-cliques
cliqueMatrix = []
cliqueMatrix.append([])
cliqueMatrix.append([])
for k in range(2, 8):
    i = 0
    cliqueMatrix.append([])
    for clique in nx.find_cliques(G):
        if len(clique) == k:
            i += 1
            cliqueMatrix[k].append(clique)
        '''elif len(clique) > k:
            i += len(list(itertools.combinations(clique, k)))'''
    print(k, "-clique: ", i)
print('\n')

# clique edges experimental


def travel_list(l, k):
    for el in l:
        a = el.get(k, None)
        if a == None:
            continue
        else:
            return a


cliqueEdges = []
target = [{str(a["source"] + a["target"]):{"s": a["source"], "d":a["target"], "tracks":','.join(
    [f["name"] + " - " + str(f["pos"]) + " - " + str(a["year"]) for f in a["tracks"]])}} for a in l_e_new]
# print("target - ", target)
for i in range(len(cliqueMatrix[6])):
    cliqueEdges.append([])
    for node1 in cliqueMatrix[6][i]:
        s = []
        for node2 in cliqueMatrix[6][i]:
            t = travel_list(target, str(node1 + node2))
            if not t in s and t is not None:
                s.append(t)
        cliqueEdges.append(list(s))

result = [c for c in cliqueEdges]  # if len(c)>1]
# print("RES - ", result)

#############################
# find how many songs in every clique
for k in range(2, 8):
    cliqueSongs = []
    songyears = 0
    songmin = 2020
    songmax = 1950
    c = 0
    for i in range(len(cliqueMatrix[k])):
        songs = []
        append_this = []
        for artist in cliqueMatrix[k][i]:
            for d in dict_vett_writers:
                if d["writer"] == artist:
                    for song in d["tracks"]:
                        songs.append(
                            song["name"] + " - " + str(song["pos"]) + " - " + str(song["year"]))
                        songyears += song["year"]
                        if song["year"] < songmin:
                            songmin = song["year"]
                        if song["year"] > songmax:
                            songmax = song["year"]
                        c += 1
            songs_set = set(songs)
            append_this = [[s, 0] for s in songs_set]
            for song in songs:
                for j in range(len(append_this)):
                    if song == append_this[j][0]:
                        append_this[j][1] += 1  # (song,append_this[j][1]+1)
            append_this = [a[0] for a in append_this if a[1] > 1]

        cliqueSongs.append("Clique number " + str(i) +
                           " - " + str(append_this))
    mediayears = songyears/c
    '''print("clique songs - ", cliqueSongs , " k = ", k)
    print("anno medio - ", mediayears, " anno minimo - ", songmin, " anno massimo - ", songmax, " k = ", k)'''


# create subgraph with all nodes and edges connected to node "D. Pace"
list_nodes_connected = list(nx.node_connected_component(G, "D. Pace"))


# calculate small world coefficient
print("small world coefficient")
print(nx.algorithms.smallworld.sigma(G.subgraph(
    list_nodes_connected), niter=10, nrand=1, seed=42), '\n')


#############################
'''plt.rcParams['figure.figsize'] = [15, 10]
#closeness centrality
print("closeness centrality")
close_centrality = nx.closeness_centrality(G)
#print(nx.closeness_centrality(G),'\n')
print(close_centrality,'\n')
#sort closeness centrality



#betweenness centrality
print("betweenness centrality")
betw_centrality = nx.betweenness_centrality(G, normalized=True,weight="weight")
print(betw_centrality,'\n')

#local clustering coefficient
print("local clustering coefficient")
local_clustering = nx.clustering(G,weight="weight")
print(local_clustering,'\n')
'''
