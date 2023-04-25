
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import *
import math


# Rotational matrix
def s(angle, rad=True):
    if rad == False:
        angle = math.radians(angle)
    return math.sin(angle)


def c(angle, rad=True):
    if rad == False:
        angle = math.radians(angle)
    return math.cos(angle)


def Rotation_BA(alpha, beta, gamma, rad=True):
    if rad == False:
        a, b, y = math.radians(alpha), math.radians(beta), math.radians(gamma)
    else:
        a, b, y = alpha, beta, gamma
    row1 = np.array([c(y) * c(b), c(y) * s(b) * s(a) - s(y) * c(a), c(y) * s(b) * c(a) + s(y) * s(a)])
    row2 = np.array([s(y) * c(b), s(y) * s(b) * s(a) + c(y) * c(a), s(y) * s(b) * c(a) - c(y) * s(a)])
    row3 = np.array([-s(b), c(b) * s(a), c(b) * c(a)])
    result = np.concatenate([row1, row2, row3]).reshape(3, 3)
    return result


def Rotation(angle, direction, rad=True):
    if rad == False:
        angle = math.radians(angle)

    if direction == 'x':
        R = np.array([[1, 0, 0],
                      [0, c(angle), -s(angle)],
                      [0, s(angle), c(angle)]])
    elif direction == 'y':
        R = np.array([[c(angle), 0, s(angle)],
                      [0, 1, 0],
                      [-s(angle), 0, c(angle)]])
    elif direction == 'z':
        R = np.array([[c(angle), -s(angle), 0],
                      [s(angle), c(angle), 0],
                      [0, 0, 1]])
    return R


def Rotation_non_pincipal_axes(angle, direction, rad=True):
    if rad == False:
        a = math.radians(angle)
    else:
        a = angle

    # Convert the direction to unit vector, if the direction is already unit vector, norm = 1
    [ux, uy, uz] = direction / np.linalg.norm(direction)

    # Use equation 33
    row1 = np.array(
        [ux * ux * (1 - c(a)) + c(a), ux * uy * (1 - c(a)) - uz * s(a), ux * uz * (1 - c(a)) + uy * s(a)]).reshape(1,
                                                                                                                   -1)
    row2 = np.array(
        [ux * uy * (1 - c(a)) + uz * s(a), uy * uy * (1 - c(a)) + c(a), uy * uz * (1 - c(a)) - ux * s(a)]).reshape(1,
                                                                                                                   -1)
    row3 = np.array(
        [ux * uz * (1 - c(a)) - uy * s(a), uy * uz * (1 - c(a)) + ux * s(a), uz * uz * (1 - c(a)) + c(a)]).reshape(1,
                                                                                                                   -1)
    result = np.concatenate([row1, row2, row3], axis=0)

    return result


def Transformation(rotation, translation):
    '''
    Transform from frame B to frame A
    rotation: rotation of B relative to A
    translation or B0: Position of origin of frame B with respect to A

    '''
    # Translation from B to A

    if np.shape(rotation) != (3, 3):
        print('Wrong rotation matrix')
        return None
    if np.shape(translation) != (3, 1):
        print('Wrong translation matrix')
        return None
    T = np.concatenate([rotation, translation], axis=1)
    add_row = np.array([0, 0, 0, 1]).reshape(1, -1)
    T = np.concatenate([T, add_row], axis=0)
    return T


def Mapping(transformation, coordinate):
    '''
    Map a point through a transformation matrix
    '''
    if np.shape(transformation) != (4, 4):
        print('Wrong transformation matrix')
        return None
    if np.shape(coordinate) != (3, 1):
        print('Wrong coordinate matrix')
        return None

    coordinate = np.append(coordinate, [[1]], axis=0)

    result = np.matmul(transformation, coordinate)

    # need to ignore the last value in the last row
    return result[:-1]


def Euler_parameters(alpha, A_U, rad=True):
    if rad == False:
        alpha = math.radians(alpha)
    [ux, uy, uz] = A_U

    e0 = ux * s(alpha / 2)
    e1 = uy * s(alpha / 2)
    e2 = uz * s(alpha / 2)
    e3 = c(alpha / 2)

    print(f'Check the summed square: {round(e0 ** 2 + e1 ** 2 + e2 ** 2 + e3 ** 2, 2)}')
    return [e0, e1, e2, e3]


def Get_angle_from_rotation(rotation):
    if np.shape(rotation) != (3, 3):
        print('Wrong rotation matrix')
        return None
    [r11, r12, r13, r21, r22, r23, r31, r32, r33] = np.squeeze(rotation.reshape(1, -1))

    # Cosine is an even function, thus we expect to have 2 answers, 1 postive and 1 negative
    alpha1 = np.arccos((r11 + r22 + r33 - 1) / 2)
    print(f'Alpha 1 is {alpha1} rad, {np.degrees(alpha1)} degrees')

    # A_U is the unit vector of revolution relative to a fixed frame {A}
    A_U_1 = 1 / (2 * s(alpha1)) * np.array([r32 - r23, r13 - r31, r21 - r12])
    print(f'Rotation axis 1 is {A_U_1}')

    euler1 = Euler_parameters(alpha1, A_U_1, rad=True)
    print(f'Euler parameter 1 is {euler1}')

    print('-----------------------------')
    alpha2 = -alpha1
    print(f'Alpha 2 is {alpha2} rad, {np.degrees(alpha2)} degrees')
    A_U_2 = 1 / (2 * s(alpha2)) * np.array([r32 - r23, r13 - r31, r21 - r12])

    print(f'Rotation axis 2 is {A_U_2}')
    euler2 = Euler_parameters(alpha2, A_U_2, rad=True)

    print(f'Euler parameter 2 is {euler2}')

    return [alpha1, A_U_1, euler1], [alpha2, A_U_2, euler2]


def Tlink(DH):
    # This function calculates the Homogeneous Transformation matrix for a
    # link, receiving as input the DH parameters for the link
    # DH parameters are in the form of an array with: link length, link
    # twist, joint offset, joint angle

    # Set: a:link length; alpha:link twist; d:joint offset; theta:joint
    # angle
    a = DH[0];
    alpha = DH[1];
    d = DH[2];
    theta = DH[3];

    # Set a 4x4 symbolic T matrix
    T = zeros(4)
    # Line 1
    T[0, 0] = cos(theta);
    T[0, 1] = -sin(theta);
    T[0, 2] = 0;
    T[0, 3] = a;
    # Line 2
    T[1, 0] = sin(theta) * cos(alpha);
    T[1, 1] = cos(theta) * cos(alpha);
    T[1, 2] = -sin(alpha);
    T[1, 3] = -sin(alpha) * d;
    # Line 3
    T[2, 0] = sin(theta) * sin(alpha);
    T[2, 1] = cos(theta) * sin(alpha);
    T[2, 2] = cos(alpha);
    T[2, 3] = cos(alpha) * d;
    # Line 4
    T[3, 0] = 0;
    T[3, 1] = 0;
    T[3, 2] = 0;
    T[3, 3] = 1;

    return T


def symprint(symbol, sup, sub, dot=False):
    if dot == 1:
        symbol = r'\dot{%s}' % symbol
    elif dot == 2:
        symbol = r'\ddot{%s}' % symbol
    if sup == '':
        info = r"{}_{}".format(symbol, sub)
    else:
        info = r"^{}{}_{}".format(sup, symbol, sub)
    display(symbols(info))


def Position_finder(sym_matrix):
    # exclude the last row and display the last column
    return sym_matrix[:-1, -1]


def revolute_joint(frame, theta_dot, transform_low_high, omega, v, Display=True, Display_all_details=False):
    # Transpose and extract the 3x3 matrix
    rotation_high_low = transform_low_high.T[:3, :3]
    P = transform_low_high[:3, -1]
    # theta_dot = symbols(r'\dot{\theta_{%s}}' % frame)
    if frame != 'e':
        omega_new = simplify(rotation_high_low * omega + theta_dot * Matrix([0, 0, 1]))
    else:
        omega_new = simplify(rotation_high_low * omega)
    v_new = simplify(rotation_high_low * (v + omega.cross(P)))

    if Display_all_details:
        print('R')
        display(rotation_high_low)
        print('v_prev')
        display(v)
        print('omega_prev')
        display(omega)
        print('P')
        display(P)
        print('omega x P')
        display(omega.cross(P))
        print('theta_dot * k')
        display(theta_dot * Matrix([0, 0, 1]))

    if Display:
        symprint('\Omega', frame, frame)
        display(omega_new)
        symprint('V', frame, frame)
        display(v_new)

    return [omega_new, v_new]


def prismatic_joint(frame, d_dot, transform_low_high, omega, v, Display=True, Display_all_details=False):
    # Transpose and extract the 3x3 matrix
    rotation_high_low = transform_low_high.T[:3, :3]
    P = transform_low_high[:3, -1]
    # d_dot = symbols(r'\dot{d_{%s}}' % frame)
    omega_new = simplify(rotation_high_low * omega)
    if frame != 'e':
        v_new = simplify(rotation_high_low * (v + omega.cross(P)) + d_dot * Matrix([0, 0, 1]))
    else:
        v_new = simplify(rotation_high_low * (v + omega.cross(P)))

    if Display_all_details:
        print('R')
        display(rotation_high_low)
        print('v_prev')
        display(v)
        print('omega_prev')
        display(omega)
        print('P')
        display(P)
        print('omega x P')
        display(omega.cross(P))
        print('d_dot * k')
        display(d_dot * Matrix([0, 0, 1]))

    if Display:
        symprint('\Omega', frame, frame)
        display(omega_new)
        symprint('V', frame, frame)
        display(v_new)

    return [omega_new, v_new]


def Jacobian(parameters, v_ee, omega_ee, transform_low_high, Display=True, Display_all_details=False):
    if len(parameters) == 3:
        a, b, c = parameters
        Jee = Matrix([[v_ee[0].diff(a), v_ee[0].diff(b), v_ee[0].diff(c)],
                      [v_ee[1].diff(a), v_ee[1].diff(b), v_ee[1].diff(c)],
                      [v_ee[2].diff(a), v_ee[2].diff(b), v_ee[2].diff(c)],
                      [omega_ee[0].diff(a), omega_ee[0].diff(b), omega_ee[0].diff(c)],
                      [omega_ee[1].diff(a), omega_ee[1].diff(b), omega_ee[1].diff(c)],
                      [omega_ee[2].diff(a), omega_ee[2].diff(b), omega_ee[2].diff(c)]])
        if Display_all_details:
            display(simplify(Matrix([[transform_low_high[:3, :3], zeros(3)], [zeros(3), transform_low_high[:3, :3]]])))
        J0 = simplify(Matrix([[transform_low_high[:3, :3], zeros(3)], [zeros(3), transform_low_high[:3, :3]]]) * Jee)

    elif len(parameters) == 4:
        a, b, c, d = parameters
        Jee = Matrix([[v_ee[0].diff(a), v_ee[0].diff(b), v_ee[0].diff(c), v_ee[0].diff(d)],
                      [v_ee[1].diff(a), v_ee[1].diff(b), v_ee[1].diff(c), v_ee[1].diff(d)],
                      [v_ee[2].diff(a), v_ee[2].diff(b), v_ee[2].diff(c), v_ee[2].diff(d)],
                      [omega_ee[0].diff(a), omega_ee[0].diff(b), omega_ee[0].diff(c), omega_ee[0].diff(d)],
                      [omega_ee[1].diff(a), omega_ee[1].diff(b), omega_ee[1].diff(c), omega_ee[1].diff(d)],
                      [omega_ee[2].diff(a), omega_ee[2].diff(b), omega_ee[2].diff(c), omega_ee[2].diff(d)]])

    if Display:
        symprint('J', 'e', 'e')
        display(Jee)
        symprint('J', 0, '')
        display(J0)

    return [Jee, J0]

if __name__ == '__main__':
    print('hello world')
