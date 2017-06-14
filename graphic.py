import matplotlib.pyplot as plt



# fig = plt.figure()
# x = [2, 3, 4, 8, 10]
# homogeneity = [0.214, 0.278, 0.345, 0.506, 0.557]
# completeness = [0.721, 0.621, 0.6, 0.601, 0.587]
# v_measure = [0.33, 0.384, 0.438, 0.549, 0.572]
# plt.plot(x, homogeneity, label='Homogeneity')
# plt.plot(x, completeness, label='Completeness')
# plt.plot(x, v_measure, label='V-measure')
# plt.xticks([i for i in range(2,11)])
# plt.ylabel('Scores')
# plt.xlabel('Clusters')
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#            ncol=2, mode="expand", borderaxespad=0.)

# plt.show()

fig = plt.figure()
x = [i for i in range(2, 21)]
indexes = [ 415.4208267847638,
            431.1077226655846,
            398.69286199695676,
            403.3371121930517,
            470.5160378198041,
            549.5164511228854,
            668.3420141543311,
            503.4515675007599,
            364.0880782547149,
            357.5542864665126,
            364.63650070597106,
            337.00121679160975,
            350.60862009743937,
            346.5536280511577,
            338.10723310521865,
            316.9037001459935,
            293.5192716318247,
            321.19020288936923,
            308.3433792475808,
]
plt.plot(x, indexes)

plt.xticks([i for i in range(2,21)])
plt.ylabel('Scores')
plt.xlabel('Clusters')
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#            ncol=2, mode="expand", borderaxespad=0.)

plt.show()