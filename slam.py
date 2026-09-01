import numpy as np


def initialize_constraints(N, num_landmarks, world_size):
    """Initialize omega and xi constraint matrices for Graph SLAM."""
    n_constraints = num_landmarks + N
    omega = np.zeros((n_constraints * 2, n_constraints * 2))
    omega[0][0] = 1
    omega[1][1] = 1

    xi = np.zeros(n_constraints * 2)
    xi[0] = world_size / 2
    xi[1] = world_size / 2
    return omega, xi


def slam(data, N, num_landmarks, world_size, motion_noise, measurement_noise):
    """Run Graph SLAM and return mu (poses and landmark positions)."""
    omega, xi = initialize_constraints(N, num_landmarks, world_size)
    one_motion_noise = 1 / motion_noise
    one_measure_noise = 1 / measurement_noise

    for i in range(len(data)):
        measurements = data[i][0]
        motion = data[i][1]

        for msr in measurements:
            lm_index = msr[0]
            x = msr[1]
            y = msr[2]

            omega[2 * i][2 * i] += one_measure_noise
            omega[2 * i + 1][2 * i + 1] += one_measure_noise
            omega[2 * N + 2 * lm_index][2 * N + 2 * lm_index] += one_measure_noise
            omega[2 * N + 2 * lm_index + 1][2 * N + 2 * lm_index + 1] += one_measure_noise

            omega[2 * i][2 * N + 2 * lm_index] += -one_measure_noise
            omega[2 * i + 1][2 * N + 2 * lm_index + 1] += -one_measure_noise
            omega[2 * N + 2 * lm_index][2 * i] += -one_measure_noise
            omega[2 * N + 2 * lm_index + 1][2 * i + 1] += -one_measure_noise

            xi[2 * i] += -x * one_measure_noise
            xi[2 * i + 1] += -y * one_measure_noise
            xi[2 * N + 2 * lm_index] += x * one_measure_noise
            xi[2 * N + 2 * lm_index + 1] += y * one_measure_noise

        disp_x = motion[0]
        disp_y = motion[1]

        omega[2 * i][2 * i] += one_motion_noise
        omega[2 * i + 1][2 * i + 1] += one_motion_noise
        omega[2 * i + 2][2 * i + 2] += one_motion_noise
        omega[2 * i + 3][2 * i + 3] += one_motion_noise

        omega[2 * i][2 * i + 2] += -one_motion_noise
        omega[2 * i + 1][2 * i + 3] += -one_motion_noise
        omega[2 * i + 2][2 * i] += -one_motion_noise
        omega[2 * i + 3][2 * i + 1] += -one_motion_noise

        xi[2 * i] += -disp_x * one_motion_noise
        xi[2 * i + 1] += -disp_y * one_motion_noise
        xi[2 * i + 2] += disp_x * one_motion_noise
        xi[2 * i + 3] += disp_y * one_motion_noise

    omega_inverse = np.linalg.inv(omega)
    mu = np.dot(omega_inverse, xi)
    return mu


def get_poses_landmarks(mu, N, num_landmarks):
    """Extract pose and landmark (x, y) pairs from the SLAM solution vector."""
    poses = []
    for i in range(N):
        poses.append((mu[2 * i].item(), mu[2 * i + 1].item()))

    landmarks = []
    for i in range(num_landmarks):
        landmarks.append((mu[2 * (N + i)].item(), mu[2 * (N + i) + 1].item()))

    return poses, landmarks


def print_all(poses, landmarks):
    """Print estimated poses and landmark locations."""
    print('\n')
    print('Estimated Poses:')
    for pose in poses:
        print('[' + ', '.join('%.3f' % p for p in pose) + ']')
    print('\n')
    print('Estimated Landmarks:')
    for landmark in landmarks:
        print('[' + ', '.join('%.3f' % value for value in landmark) + ']')
