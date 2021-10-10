import numpy as np
import cvxopt as cvx


bigfinger_vertices = [697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707,
                      708, 709, 710, 711, 712, 713, 714,
                      715, 716, 717, 718, 719, 720, 721, 722, 723, 724,
                      725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737,
                      738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750,
                      751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763,
                      764, 765, 766, 767, 768]

indexfinger_vertices = [46, 47, 48, 49, 56, 57, 58, 59, 86, 87, 133, 134, 155,
                        156, 164, 165, 166, 167, 174, 175, 189, 194, 195, 212,
                        213, 221, 222, 223, 224, 225, 226, 237, 238, 272, 273,
                        280, 281, 282, 283, 294, 295, 296, 297, 298, 299, 300,
                        301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
                        312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322,
                        323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333,
                        334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345,
                        346, 347, 348, 349, 350, 351, 352, 353, 354, 355]

middlefinger_vertices = [356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367,
                         372, 373, 374, 375, 376, 377, 381,
                         382, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394,
                         395, 396, 397, 398, 400, 401, 402, 403, 404, 405, 406, 407,
                         408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420,
                         421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433,
                         434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446,
                         447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459,
                         460, 461, 462, 463, 464, 465, 466, 467]

fourthfinger_vertices = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,
                         482, 483, 484, 485, 486, 487, 491, 492,
                         495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507,
                         508, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520,
                         521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533,
                         534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546,
                         547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559,
                         560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572,
                         573, 574, 575, 576, 577, 578]

smallfinger_vertices = [580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591,
                        598, 599, 600, 601, 602, 603,
                        609, 610, 613, 614, 615, 616, 617, 618, 619, 620,
                        621, 622, 623, 624, 625, 626, 628, 629, 630, 631, 632, 633,
                        634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646,
                        647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659,
                        660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672,
                        673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685,
                        686, 687, 688, 689, 690, 691, 692, 693, 694, 695]


def graspit_measure(forces, torques, normals):
    G = grasp_matrix(np.array(forces).transpose(), np.array(torques).transpose(), np.array(normals).transpose())
    measure = min_norm_vector_in_facet(G)[0]
    return measure


def get_normal_face(p1, p2, p3):
    U = p2 - p1
    V = p3 - p1
    Nx = U[1]*V[2] - U[2]*V[1]
    Ny = U[2]*V[0] - U[0]*V[2]
    Nz = U[0]*V[1] - U[1]*V[0]
    return [Nx, Ny, Nz]


def get_distance_vertices(obj, hand):
    n1 = len(hand)
    n2 = len(obj)
    matrix1 = hand[np.newaxis].repeat(n2, 0)
    matrix2 = obj[:, np.newaxis].repeat(n1, 1)
    dists = np.sqrt(((matrix1-matrix2)**2).sum(-1))
    return dists.min(0)


def grasp_matrix(forces, torques, normals, soft_fingers=False,
                 finger_radius=0.005, params=None):
    if params is not None and 'finger_radius' in params.keys():
        finger_radius = params.finger_radius
    num_forces = forces.shape[1]
    num_torques = torques.shape[1]
    if num_forces != num_torques:
        raise ValueError('Need same number of forces and torques')

    num_cols = num_forces
    if soft_fingers:
        num_normals = 2
        if normals.ndim > 1:
            num_normals = 2*normals.shape[1]
        num_cols = num_cols + num_normals

    torque_scaling = 1
    G = np.zeros([6, num_cols])
    for i in range(num_forces):
        G[:3,i] = forces[:,i]
        G[3:,i] = torque_scaling * torques[:,i]

    if soft_fingers:
        torsion = np.pi * finger_radius**2 * params.friction_coef * normals * params.torque_scaling
        pos_normal_i = -num_normals
        neg_normal_i = -num_normals + num_normals / 2
        G[3:,pos_normal_i:neg_normal_i] = torsion
        G[3:,neg_normal_i:] = -torsion

    return G


def get_contact_points(hand_verts, hand_faces, obj_verts):
    finger_vertices = [indexfinger_vertices, middlefinger_vertices, fourthfinger_vertices,
                       smallfinger_vertices, bigfinger_vertices]
    forces = []
    torques = []
    normals = []
    finger_is_touching = np.zeros(5)
    threshold = 0.004  # m
    for i in range(len(finger_vertices)):
        dists = get_distance_vertices(obj_verts, hand_verts[finger_vertices[i]])
        if np.min(dists) < threshold:
            finger_is_touching[i] = 1
            faces = np.where(finger_vertices[i][np.argmin(dists)] == hand_faces)[0]
            normal = []
            for j in range(len(faces)):
                normal.append(get_normal_face(hand_verts[hand_faces[faces[j], 0]], hand_verts[hand_faces[faces[j], 1]],
                                              hand_verts[hand_faces[faces[j], 2]]))
            normal = np.mean(normal, 0) * 1e5  # Multiply by large number to avoid **2 going to zero
            normal = normal / np.sqrt((np.array(normal) ** 2).sum())
            torques.append([0, 0, 0])
            normals.append(normal)
            forces.append(normal)

    return forces, torques, normals, finger_is_touching


def min_norm_vector_in_facet(facet, wrench_regularizer=1e-10):
    """ Finds the minimum norm point in the convex hull of a given facet (aka simplex) by solving a QP.
    Parameters
    ----------
    facet : 6xN :obj:`numpy.ndarray`
        vectors forming the facet
    wrench_regularizer : float
        small float to make quadratic program positive semidefinite
    Returns
    -------
    float
        minimum norm of any point in the convex hull of the facet
    Nx1 :obj:`numpy.ndarray`
        vector of coefficients that achieves the minimum
    """
    dim = facet.shape[1] # num vertices in facet

    # create alpha weights for vertices of facet
    G = facet.T.dot(facet)
    grasp_matrix = G + wrench_regularizer * np.eye(G.shape[0])

    # Solve QP to minimize .5 x'Px + q'x subject to Gx <= h, Ax = b
    P = cvx.matrix(2 * grasp_matrix)   # quadratic cost for Euclidean dist
    q = cvx.matrix(np.zeros((dim, 1)))
    G = cvx.matrix(-np.eye(dim))       # greater than zero constraint
    h = cvx.matrix(np.zeros((dim, 1)))
    A = cvx.matrix(np.ones((1, dim)))  # sum constraint to enforce convex
    b = cvx.matrix(np.ones(1))         # combinations of vertices

    sol = cvx.solvers.qp(P, q, G, h, A, b)
    v = np.array(sol['x'])
    min_norm = np.sqrt(sol['primal objective'])

    return abs(min_norm), v



