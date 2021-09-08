import numpy as np
import io

# need to change to own dir
f = io.open('/home/wayne/github/Pulp_Fiction_non_words.1D', mode="r", encoding="utf-8")
f2 = io.open('/home/wayne/github/pulp_fiction_nonwordsounds.1D', mode="r", encoding="utf-8")

# for opening the data set and separate them with `:`
f_non_words = np.loadtxt(f, delimiter = ':')
f_sounds = np.loadtxt(f2, delimiter=':')

# calculating the endpoints for each data point
for x in range(np.shape(f_non_words)[0]):
    f_non_words[x, 1] = f_non_words[x,0] + f_non_words[x, 1]

for x in range(np.shape(f_sounds)[0]):
    f_sounds[x, 1] = f_sounds[x,0] + f_sounds[x, 1]   

# to make 2d array to 1d
f_non_words = f_non_words.flatten()
f_sounds = f_sounds.flatten()


# to include x(n-1), y(n), y(n+1), x(n) as the new list for *odd* index in f_sounds
list = np.array([])
for y in range(0, len(f_sounds), 2):
    tmp_matrix = np.zeros(4)
    tmp_ind = np.argwhere(f_non_words > f_sounds[y])[0]
    tmp_matrix = [f_non_words[int(tmp_ind)-1], f_sounds[y], f_sounds[y+1], f_non_words[int(tmp_ind)]]
    list = np.append(list, tmp_matrix)

# calculate the interval
new_interval = np.array([])
for x in range(0, len(list), 2):
    tmp = list[x+1] - list[x]
    new_interval = np.append(new_interval, tmp)





