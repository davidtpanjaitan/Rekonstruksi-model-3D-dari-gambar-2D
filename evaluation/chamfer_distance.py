from math import sqrt
import numpy as np
import pywavefront as pwf

def euclidean_distance(p1, p2):
    return sqrt(sum([(x2-x1)**2 for x1,x2 in zip(p1, p2)]))

def chamfer_distance(points1, points2):
    distance = 0
    # if (len(points1)*len(points2) < 6e8):
    #     print('calculating distances: ', end='')
    #     distances = np.linalg.norm(points1-points2[:,None], axis=-1)
    #     print('done!')
    #     print('calculating CD: ', end='')
    #     distance = np.sum(np.min(distances, axis=0)) + np.sum(np.min(distances, axis=1))
    #     print('done!')
    #     return distance
    # else:
    print('vertices too many, calculating using memory saving mode: ')
    for i in range(len(points1)):
        distance += np.min(np.linalg.norm([points2-points1[i]], axis=-1))
    for i in range(len(points2)):
        distance += np.min(np.linalg.norm([points1-points2[i]], axis=-1))
    
    return distance

def chamfer_distance_3D_obj(obj1, obj2, normalized=True):
    print('loading obj files: ', end='')
    v_list1 = get_v_list_from_file(obj1, normalized, obj1+'.txt')
    v_list2 = get_v_list_from_file(obj2, normalized, obj2+'.txt')
    print('done')
    return chamfer_distance(v_list1, v_list2), (len(v_list1), len(v_list2))

def get_v_list_from_file(f, normalized=True, name='temp.txt'):
    fobj = pwf.Wavefront(f)
    v_list = np.array(fobj.vertices, dtype='float32')[:,:3]
    
    # with open(f, 'r') as fobj:
    #     v_list = []
    #     line = fobj.readline().split()
    #     while line[0] != 'v':
    #         line = fobj.readline().split()
    #     while line[0] == 'v':
    #         v_list.append([float(line[1]), float(line[2]), float(line[3])])
    #         line = fobj.readline().split()
    # v_list = np.array(v_list)
    if normalized:
        maxp = np.max(v_list, axis=0)
        minp = np.min(v_list, axis=0)
        normalized = (v_list - minp) / (maxp[1] - minp[1]) #np.linalg.norm(v_list)
        print_vlist(normalized, name)
        return normalized
    else:
        return v_list

def print_vlist(v_list, name='temp.txt'):
    with open(name, 'w') as of:
        for p in v_list:
            print('v', p[0], p[1], p[2], sep=' ', file=of)

if __name__ == '__main__':
    # fobj = pwf.Wavefront('carla.obj')
    # v_list = get_v_list_from_file('carla.obj') #np.array(fobj.vertices)[:,:3]
    # print(v_list.shape)
    # print('max0', np.max(v_list, axis=0))
    # print('max1', np.max(v_list, axis=1))
    # print('min0', np.min(v_list, axis=0))
    # print('min1', np.min(v_list, axis=1))
    # print('writing to file')

    base = ['eric', 'sophia', 'carla', 'dennis', 'claudia']
    comps = []
    for w in base:
        comps.append([w+'.obj', 'result_'+w+'_a.obj'])
        comps.append([w+'.obj', 'result_'+w+'_m.obj'])
    print(comps)
    comps.append(['eric.obj', 'result_multi_eric.obj'])
    comps.append(['dennis.obj', 'result_multi_dennis.obj'])
    comps.append(['sophia.obj', 'result_multi_sophia.obj'])
    results = []
    for i, (obj1, obj2) in enumerate(comps):
        # obj1, obj2 = input('enter the 2 file names:').split()
        print('comparison ',i, ' of ',len(comps),':', obj1, ' - ', obj2)
        r, (n1, n2) = chamfer_distance_3D_obj(obj1, obj2)
        print(r)
        results.append(r)
        comps[i].append([n1,n2])
    with open('log.txt', 'w') as f:
        for i in range(len(comps)):
            print(comps[i], ' = ', results[i], ' / ', results[i]/np.sum(comps[i][2]), file=f)
    
