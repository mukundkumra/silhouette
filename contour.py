import numpy as np


def find_contours(edges):
    found_contours = []
    evaluated_matrix = np.full(edges.shape, False)

    for y in range(0, edges.shape[0]):
        for x in range(0, edges.shape[1]):
            if edges[y, x] == 0 or evaluated_matrix[y, x] == True:
                pass
            else:
                contour, evaluated_matrix = find_consecutive(edges, evaluated_matrix, (y, x))
                if len(contour) > 0:
                    found_contours.append(contour)
    return np.array(found_contours)
                

def find_consecutive(edges, evaluated_matrix, origin):
    contour_points = []
    next_points = [origin]

    while next_points:
        y, x = next_points.pop()
        if (edges[y, x] != 0 and evaluated_matrix[y, x] == False):
            neighbors = [(y+1, x), (y-1, x), (y, x+1), (y, x-1)]
            for y2, x2 in neighbors:
                if 0 <= y2 < edges.shape[0] and 0 <= x2 < edges.shape[1]:
                    if evaluated_matrix[y2, x2] == False:
                        next_points.append((y2, x2))
            contour_points.append(np.array((x, y)))
        evaluated_matrix[y, x] = True
    return np.array(contour_points), evaluated_matrix


def fill_contours(contour, edges):
    """Should return area as well as filled contour"""
    for x, y in contour:
        point = (x+1, y)
        if point[0] >= edges.shape[1]:
            continue
        while edges[point[1], point[0]] == 0:
            pass
