"""
@Desc:
@Reference:
- networkx+python构建图结构数据并可视化
https://blog.csdn.net/leviopku/article/details/107018547
@Notes:
"""




import networkx as nx
import matplotlib.pyplot as plt

g = nx.Graph()
g.add_edge('1', '2')
g.add_edge('2', '3')
g.add_edge('1', '4')
g.add_edge('2', '4')

fixed_position = {'1': [1, 1], '2': [1.5, 0.8], '3': [1.7, 2.8], '4': [0.6, 3.3]}
pos = nx.spring_layout(g, pos=fixed_position)

colors = []
for i in range(g.number_of_nodes()):
    if i == 2:
        colors.append('#ff0000')
    else:
        colors.append('#1f7814')

fig, ax = plt.subplots()
nx.draw(g, ax=ax, with_labels=True, pos=pos, node_color=colors)  # add colors
plt.show()