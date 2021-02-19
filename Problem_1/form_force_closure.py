import cvxpy as cp
import numpy as np

def cross_matrix(x):
    """
    Returns a matrix x_cross such that x_cross.dot(y) represents the cross
    product between x and y.

    For 3D vectors, x_cross is a 3x3 skew-symmetric matrix. For 2D vectors,
    x_cross is a 2x1 vector representing the magnitude of the cross product in
    the z direction.
     """
    D = x.shape[0]
    if D == 2:
        return np.array([[-x[1], x[0]]])
    elif D == 3:
        return np.array([[0., -x[2], x[1]],
                         [x[2], 0., -x[0]],
                         [-x[1], x[0], 0.]])
    raise RuntimeError("cross_matrix(): x must be 2D or 3D. Received a {}D vector.".format(D))

def wrench(f, p):
    """
    Computes the wrench from the given force f applied at the given point p.
    Works for 2D and 3D.

    Args:
        f - 2D or 3D contact force.
        p - 2D or 3D contact point.

    Return:
        w - 3D or 6D contact wrench represented as (force, torque).
    """
    ########## Your code starts here ##########
    # Hint: you may find cross_matrix(x) defined above helpful. This should be one line of code.
    w = np.hstack((f, np.dot(cross_matrix(p), f)))
    ########## Your code ends here ##########

    return w

def cone_edges(f, mu):
    """
    Returns the edges of the specified friction cone. For 3D vectors, the
    friction cone is approximated by a pyramid whose vertices are circumscribed
    by the friction cone.

    In the case where the friction coefficient is 0, a list containing only the
    original contact force is returned.

    Args:
        f - 2D or 3D contact force.
        mu - friction coefficient.

    Return:
        edges - a list of forces whose convex hull approximates the friction cone.
    """
    # Edge case for frictionless contact
    if mu == 0.:
        return [f]

    # Planar wrenches
    D = f.shape[0]
    if D == 2:
        ########## Your code starts here ##########
        edges = [np.zeros(D)] * 2

        # get the perpendicular to our force vector
        f_perp = np.array([-f[1], f[0]])

        # calculate the first edge
        edges[0] = f + mu * f_perp / np.linalg.norm(f_perp)

        # calculate the second edge
        edges[1] = f - mu * f_perp / np.linalg.norm(f_perp)
        ########## Your code ends here ##########

    # Spatial wrenches
    elif D == 3:
        ########## Your code starts here ##########
        edges = [np.zeros(D)] * 4

        # create a random vector that we will find our first f_perp to
        rand_vec = np.random.rand(3, 1).flatten()

        # get the first perpendicular vector
        f_perp = np.dot(cross_matrix(rand_vec), f)

        # get the second perpendicular vector
        f_perp2 = np.dot(cross_matrix(f), f_perp)

        # get the first edge
        edges[0] = f + mu * f_perp / np.linalg.norm(f_perp)

        # get the second edge
        edges[1] = f - mu * f_perp / np.linalg.norm(f_perp)

        # get the third edge
        edges[2] = f + mu * f_perp2 / np.linalg.norm(f_perp2)

        # get the fourth edge
        edges[3] = f - mu * f_perp2 / np.linalg.norm(f_perp2)
        ########## Your code ends here ##########

    else:
        raise RuntimeError("cone_edges(): f must be 3D or 6D. Received a {}D vector.".format(D))

    return edges

def form_closure_program(F):
    """
    Solves a linear program to determine whether the given contact wrenches
    are in form closure.

    Args:
        F - matrix whose columns are 3D or 6D contact wrenches.

    Return:
        True/False - whether the form closure condition is satisfied.
    """
    ########## Your code starts here ##########
    # Hint: You may find np.linalg.matrix_rank(F) helpful
    # first check if F is full rank
    if np.linalg.matrix_rank(F) != min(F.shape):
        return False

    # Setup the convex optimization problem
    k = cp.Variable((F.shape[1], 1))
    ones_vec = np.ones((1, F.shape[1]))
    objective = cp.Minimize(cp.sum_squares(ones_vec @ k))
    constraints = [k >= 1, F @ k == 0]
    ########## Your code ends here ##########

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)

    return prob.status not in ['infeasible', 'unbounded']

def is_in_form_closure(forces, points):
    """
    Calls form_closure_program() to determine whether the given contact forces
    are in form closure.

    Args:
        forces - list of 2D or 3D contact forces.
        points - list of 2D or 3D contact points.

    Return:
        True/False - whether the forces are in form closure.
    """
    ########## Your code starts here ##########
    # for each force, construct our wrench
    wrenches = [wrench(forces[i], points[i]) for i in range(len(forces))]

    # Construct the F matrix of proper dimension
    F = np.zeros((len(wrenches[0]), len(forces)))

    # add each wrench to a column of F
    for idx, wr in enumerate(wrenches):
        F[:, idx] = wr

    ########## Your code ends here ##########

    return form_closure_program(F)

def is_in_force_closure(forces, points, friction_coeffs):
    """
    Calls form_closure_program() to determine whether the given contact forces
    are in force closure.

    Args:
        forces - list of 2D or 3D contact forces.
        points - list of 2D or 3D contact points.
        friction_coeffs - list of friction coefficients.

    Return:
        True/False - whether the forces are in force closure.
    """
    ########## Your code starts here ##########
    # check if 2D or 3D
    is_2D = len(forces[0]) == 2
    # first check number of rows
    rows = 3 if is_2D else 6

    # then calculate the number of columns assuming all contact points have friction
    cols = len(forces) * 2 if is_2D else len(forces) * 4

    # then remove 1 columns for every frictionless contact we have if 2D, otherwise remove 3 columns for each
    scale_factor = 1 if is_2D else 3
    cols -= sum(mu == 0 for mu in friction_coeffs) * scale_factor

    # Construct the F matrix of proper dimension
    F = np.zeros((rows, cols))

    # create an iterator
    curr_col = 0

    # for each force, find the edges
    for idx, force in enumerate(forces):
        # get the edges of the force
        edges = cone_edges(force, friction_coeffs[idx])

        # for each edge, find the wrench
        for edge in edges:
            wr = wrench(edge, points[idx])

            # add the wrench to our force matrix
            F[:, curr_col] = wr
            curr_col += 1

    ########## Your code ends here ##########

    return form_closure_program(F)
