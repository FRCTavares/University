def vector_add(v1, v2):
    return [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]]

def vector_subtract(v1, v2):
    return [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]

def vector_scale(v, scalar):
    return [v[0] * scalar, v[1] * scalar, v[2] * scalar]

def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

def cross_product(v1, v2):
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    ]

def normalize(v):
    length = math.sqrt(dot_product(v, v))
    if length == 0:
        return [0, 0, 0]
    return vector_scale(v, 1.0 / length)

def distance(v1, v2):
    return math.sqrt(dot_product(vector_subtract(v1, v2), vector_subtract(v1, v2)))