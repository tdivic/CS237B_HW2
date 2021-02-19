#!/usr/bin/env python

import cvxpy as cp
import numpy as np

from utils import *

def solve_socp(x, As, bs, cs, ds, F, g, h, verbose=False):
    """
    Solves an SOCP of the form:

    minimize(h^T x)
    subject to:
        ||A_i x + b_i||_2 <= c_i^T x + d_i    for all i
        F x == g

    Args:
        x       - cvx variable.
        As      - list of A_i numpy matrices.
        bs      - list of b_i numpy vectors.
        cs      - list of c_i numpy vectors.
        ds      - list of d_i numpy vectors.
        F       - numpy matrix.
        g       - numpy vector.
        h       - numpy vector.
        verbose - whether to print verbose cvx output.

    Return:
        x - the optimal value as a numpy array, or None if the problem is
            infeasible or unbounded.
    """
    objective = cp.Minimize(h.T @ x)
    constraints = []
    for A, b, c, d in zip(As, bs, cs, ds):
        constraints.append(cp.SOC(c.T @ x + d, A @ x + b))
    constraints.append(F @ x == g)
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)

    if prob.status in ['infeasible', 'unbounded']:
        return None

    return x.value

def grasp_optimization(grasp_normals, points, friction_coeffs, wrench_ext):
    """
    Solve the grasp force optimization problem as an SOCP. Handles 2D and 3D cases.

    Args:
        grasp_normals   - list of M surface normals at the contact points, pointing inwards.
        points          - list of M grasp points p^(i).
        friction_coeffs - friction coefficients mu_i at each point p^(i).
        wrench_ext      - external wrench applied to the object.

    Return:
        f - grasp forces as a list of M numpy arrays.
    """
    D = points[0].shape[0]  # planar: 2, spatial: 3
    N = wrench_size(D)      # planar: 3, spatial: 6
    M = len(points)
    transformations = [compute_local_transformation(n) for n in grasp_normals]

    ########## Your code starts here ##########
    # extract the list of transformations into a transformation matrix
    transformations_mat = np.zeros((D, len(transformations) * transformations[0].shape[1]))
    for idx, T in enumerate(transformations):
        transformations_mat[:, idx * D:idx * D + D] = T

    # get the P skew-symmetric cross matrices
    Ps = [cross_matrix(p) for p in points]

    # create the bottom row of the grasp map
    Phi_bot_row = np.zeros((Ps[0].shape[0], len(Ps) * Ps[0].shape[1]))
    for idx, P in enumerate(Ps):
        Phi_bot_row[:, idx * D:idx * D + D] = np.matmul(P, transformations[idx])

    # create the grasp map
    Phi = np.vstack((transformations_mat, Phi_bot_row))

    # create the F matrix
    Phi_plus = np.hstack((Phi, np.zeros((Phi.shape[0], 1))))
    F = np.vstack((Phi_plus, np.zeros((1, Phi_plus.shape[1]))))

    # create the g vector
    g = np.append(-wrench_ext, 0).T

    # create the h vector
    h = np.append(np.zeros((1, D*M)), 1).T

    # define all the A, b, c, and d lists
    As = []
    bs = []
    cs = []
    ds = []

    # loop through all the M points to define the constraint matrices
    for i in range(M):
        # add the first constraint (the second order cone constraint) matrices
        bs.append(np.zeros((D, 1)))
        cs.append(np.append(np.zeros((1, D*M)), 1).T)
        ds.append(0)

        # for the A, first create an all-zeros matrix
        A_i = np.zeros((D, D*M + 1))

        # then place the identity matrix in the proper spot and add to the list
        A_i[:, i*D:(i*D + D)] = np.eye(D)
        As.append(A_i)

        # now add the second constraint (no slip constraint) matrices
        bs.append(np.zeros((D, 1)))
        ds.append(0)

        # create the specialized c by placing a mu at the z component of each force
        c = np.zeros(D*M + 1)
        c[i*D + D - 1] = friction_coeffs[i]
        cs.append(c)

        # for the A matrix, create a modified identity matrix, place in the right spot, then add to the list
        I_mod = np.eye(D)
        I_mod[-1, -1] = 0
        A_i_slip = np.zeros((D, D*M + 1))
        A_i_slip[:, i * D:(i * D + D)] = I_mod
        As.append(A_i_slip)

    # create the x variable with the proper shape
    x = cp.Variable(D*M + 1)

    x = solve_socp(x, As, bs, cs, ds, F, g, h, verbose=False)

    # extract the grasp forces from x as a stacked 1D vector
    f = x[:-1]
    ########## Your code ends here ##########

    # Transform the forces to the global frame
    F = f.reshape(M,D)
    forces = [T.dot(f) for T, f in zip(transformations, F)]

    return forces

def precompute_force_closure(grasp_normals, points, friction_coeffs):
    """
    Precompute the force optimization problem so that force closure grasps can
    be found for any arbitrary external wrench without redoing the optimization.

    Args:
        grasp_normals   - list of M surface normals at the contact points, pointing inwards.
        points          - list of M grasp points p^(i).
        friction_coeffs - friction coefficients mu_i at each point p^(i).

    Return:
        force_closure(wrench_ext) - a function that takes as input an external wrench and
                                    returns a set of forces that maintains force closure.
    """
    D = points[0].shape[0]  # planar: 2, spatial: 3
    N = wrench_size(D)      # planar: 3, spatial: 6
    M = len(points)

    ########## Your code starts here ##########
    # Precompute the optimal forces for the 12 signed unit external
    # wrenches and store them as rows in the matrix F. This matrix will be
    # captured by the returned force_closure() function.
    F = np.zeros((2*N, M*D))

    # iterate to get our unit vectors and optimal forces
    for i in range(N):
        # create the wrench unit vectors
        w_i_plus = np.zeros(N)
        w_i_plus[i] = 1
        w_i_neg = -w_i_plus

        # solve for the optimal forces for each wrench unit
        forces_i_plus = grasp_optimization(grasp_normals, points, friction_coeffs, w_i_plus)
        force_i_neg = grasp_optimization(grasp_normals, points, friction_coeffs, w_i_neg)

        # add to the F matrix
        F[i, :] = np.array(forces_i_plus).flatten()
        F[i+N, :] = np.array(force_i_neg).flatten()

    ########## Your code ends here ##########

    def force_closure(wrench_ext):
        """
        Return a set of forces that maintain force closure for the given
        external wrench using the precomputed parameters.

        Args:
            wrench_ext - external wrench applied to the object.

        Return:
            f - grasp forces as a list of M numpy arrays.
        """

        ########## Your code starts here ##########
        # Compute the force closure forces as a stacked vector of shape (D*M)
        f = np.zeros(M*D)

        # loop through the elements of wrench_ext
        for i in range(len(wrench_ext)):
            # extract the positive and negative parts of w_i
            w_i_pos = max(0, wrench_ext[i])
            w_i_neg = max(0, -wrench_ext[i])

            # extract the positive and negative optimal forces for the unit vector
            f_i_pos = F[i, :]
            f_i_neg = F[i+N, :]

            # add the components to our f vector
            f += np.dot(w_i_pos, f_i_pos) + np.dot(w_i_neg, f_i_neg)

        ########## Your code ends here ##########

        forces = [f_i for f_i in f.reshape(M,D)]
        return forces

    return force_closure
