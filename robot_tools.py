
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
        R = Matrix([[1, 0, 0],
                    [0, cos(angle), -sin(angle)],
                    [0, sin(angle), cos(angle)]])
    elif direction == 'y':
        R = Matrix([[cos(angle), 0, sin(angle)],
                    [0, 1, 0],
                    [-sin(angle), 0, cos(angle)]])
    elif direction == 'z':
        R = Matrix([[cos(angle), -sin(angle), 0],
                    [sin(angle), cos(angle), 0],
                    [0, 0, 1]])
    return R


def Rotation_non_pincipal_axes(angle, direction, rad=True):
    if rad == False:
        a = math.radians(angle)
    else:
        a = angle

    # Convert the direction to unit vector, if the direction is already unit vector, norm = 1
    [ux, uy, uz] = direction

    # Use equation 33
    row1 = Matrix([ux * ux * (1 - cos(a)) + cos(a),
                   ux * uy * (1 - cos(a)) - uz * sin(a),
                   ux * uz * (1 - cos(a)) + uy * sin(a)]).T

    row2 = Matrix([ux * uy * (1 - cos(a)) + uz * sin(a),
                   uy * uy * (1 - cos(a)) + cos(a),
                   uy * uz * (1 - cos(a)) - ux * sin(a)]).T

    row3 = Matrix([ux * uz * (1 - cos(a)) - uy * sin(a),
                   uy * uz * (1 - cos(a)) + ux * sin(a),
                   uz * uz * (1 - cos(a)) + cos(a)]).T

    full_mat = simplify(Matrix([row1, row2, row3]))

    return full_mat


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

    coordinate = coordinate.row_insert(3, Matrix([1]))
    result = transformation * coordinate

    # need to ignore the last value in the last row
    return result


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


def matprint(matrix, alias=None):
    if alias:
        display(matrix.subs(alias))
    else:
        display(matrix)

def Position_finder(sym_matrix):
    # exclude the last row and display the last column
    return sym_matrix[:-1, -1]


def revolute_joint(frame, theta_dot, transform_low_high, omega, v, alias, Display=True, Display_all_details=False):
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
        matprint(rotation_high_low, alias)
        print('v_prev')
        matprint(v, alias)
        print('omega_prev')
        matprint(omega, alias)
        print('P')
        matprint(P, alias)
        print('omega x P')
        matprint(omega.cross(P), alias)
        print('theta_dot * k')
        matprint(theta_dot * Matrix([0, 0, 1]), alias)

    if Display:
        symprint('\Omega', frame, frame)
        matprint(omega_new, alias)
        symprint('V', frame, frame)
        matprint(v_new, alias)

    return [omega_new, v_new]


def prismatic_joint(frame, d_dot, transform_low_high, omega, v, alias, Display=True, Display_all_details=False):
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
        matprint(rotation_high_low, alias)
        print('v_prev')
        matprint(v, alias)
        print('omega_prev')
        matprint(omega, alias)
        print('P')
        matprint(P, alias)
        print('omega x P')
        matprint(omega.cross(P), alias)
        print('d_dot * k')
        matprint(d_dot * Matrix([0, 0, 1]), alias)

    if Display:
        symprint('\Omega', frame, frame)
        matprint(omega_new, alias)
        symprint('V', frame, frame)
        matprint(v_new, alias)

    return [omega_new, v_new]


def Jacobian(parameters, v_ee, omega_ee, transform_low_high, alias, Display=True, Display_all_details=False):
    Jee = Matrix([[v_ee[0].diff(var) for var in parameters],
                  [v_ee[1].diff(var) for var in parameters],
                  [v_ee[2].diff(var) for var in parameters],
                  [omega_ee[0].diff(var) for var in parameters],
                  [omega_ee[1].diff(var) for var in parameters],
                  [omega_ee[2].diff(var) for var in parameters]])

    if Display_all_details:
        matprint(simplify(Matrix([[transform_low_high[:3, :3], zeros(3)], [zeros(3), transform_low_high[:3, :3]]])), alias)
    J0 = simplify(Matrix([[transform_low_high[:3, :3], zeros(3)], [zeros(3), transform_low_high[:3, :3]]]) * Jee)

    if Display:
        symprint('J', 'e', 'e')
        matprint(Jee, alias)
        symprint('J', 0, '')
        matprint(J0, alias)

    return [Jee, J0]


def velocity_COM(frame, transform_low_high, P, v, omega, alias):
    rotation_high_low = transform_low_high.T[:3, :3]

    symprint('Omega_G', frame, frame)
    OmegaGee = simplify(rotation_high_low * omega)
    matprint(OmegaGee, alias)

    symprint('V_G', frame, frame)
    VGee = simplify(v + omega.cross(P))
    matprint(VGee, alias)

    return OmegaGee, VGee


def image_projection(w, h, nx, ny, f, coord_universe, T0C, Display=False):
    '''
    w: width of retina in m
    h: height of retina in m
    nx: horizontal pixels number
    ny: vertical pixels number
    f: focal length in m
    coord_universe: list -> Coordinate of the object / point in the universe frame
    TOC: Sympy Matrix -> Position of camera relative to the universe

    '''

    [xp, yp, zp] = coord_universe

    # Size of te individual pixel
    Dw = symbols('dw')
    Dh = symbols('dh')

    # Principal point (centre of the renita)
    U0 = symbols('u0')
    V0 = symbols('v0')

    # Focal length in m
    F = symbols('f')

    # Coordinate of the object / point in the universe frame
    Xp = symbols('x_p')
    Yp = symbols('y_p')
    Zp = symbols('z_p')

    # Coordinate of the object homogeneous form
    P0 = Matrix([Xp, Yp, Zp, 1])

    # Camera parameters
    CM = Matrix([[F / Dw, 0, U0, 0],
                 [0, F / Dh, V0, 0],
                 [0, 0, 1, 0]])

    RES = CM * T0C ** -1 * P0

    # Retina coordinate homogeneous form
    homo_retina = RES.subs({U0: nx / 2, V0: ny / 2,
                            F: f,
                            Dw: w / nx, Dh: h / ny,
                            Xp: xp, Yp: yp, Zp: zp})

    # Retina coordinate
    retina = Matrix([round(homo_retina[0] / homo_retina[-1], 1),
                     round(homo_retina[1] / homo_retina[-1], 1)])

    if Display:
        print('Camera parameters matrix')
        display(CM.subs({F: f, Dw: w / nx, Dh: h / ny, U0: nx / 2, V0: ny / 2}))

        print('Position of camera to the universe')
        display(T0C)

        print('Position of the object in the universe')
        display(P0.subs({Xp: xp, Yp: yp, Zp: zp}))

        print('Retina coordinates homogeneous')
        display(homo_retina)

        print('Retina coordinates')
        display(retina)

    # Retina coordinate
    u, v = int(retina[0]), int(retina[1])
    return [u, v]

if __name__ == '__main__':
    print('hello world')
