import numpy as np
from scipy.stats import norm

def lerp(val, low, high):
    """Linear interpolation"""
    return low + (high - low) * val

def lerp_gaussian(val, low, high):
    """Linear interpolation with gaussian CDF"""
    low_gau = norm.cdf(low)
    high_gau = norm.cdf(high)
    lerped_gau = lerp(val, low_gau, high_gau)
    return norm.ppf(lerped_gau)

def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif np.allclose(low, high):
        return low
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high

def slerp_gaussian(val, low, high):
    """Spherical interpolation with gaussian CDF (generally not useful)"""
    offset = norm.cdf(np.zeros_like(low))  # offset is just [0.5, 0.5, ...]
    low_gau_shifted = norm.cdf(low) - offset
    high_gau_shifted = norm.cdf(high) - offset
    circle_lerped_gau = slerp(val, low_gau_shifted, high_gau_shifted)
    epsilon = 0.001
    clipped_sum = np.clip(circle_lerped_gau + offset, epsilon, 1.0 - epsilon)
    result = norm.ppf(clipped_sum)
    return result

def get_interpfn(spherical, gaussian):
    """Returns an interpolation function"""
    if spherical and gaussian:
        return slerp_gaussian
    elif spherical:
        return slerp
    elif gaussian:
        return lerp_gaussian
    else:
        return lerp

def create_mine_grid(rows, cols, dim, space, anchors, spherical, gaussian, scale=1.):
    """Create a grid of latents with splash layout"""
    lerpv = get_interpfn(spherical, gaussian)

    u_list = np.zeros((rows, cols, dim))
    # compute anchors
    cur_anchor = 0
    for y in range(rows):
        for x in range(cols):
            if y%space == 0 and x%space == 0:
                if anchors is not None and cur_anchor < len(anchors):
                    u_list[y,x,:] = anchors[cur_anchor]
                    cur_anchor = cur_anchor + 1
                else:
                    u_list[y,x,:] = np.random.normal(0,scale, (1, dim))
    # interpolate horizontally
    for y in range(rows):
        for x in range(cols):
            if y%space == 0 and x%space != 0:
                lastX = space * (x // space)
                nextX = lastX + space
                fracX = (x - lastX) / float(space)
#                 print("{} - {} - {}".format(lastX, nextX, fracX))
                u_list[y,x,:] = lerpv(fracX, u_list[y, lastX, :], u_list[y, nextX, :])
    # interpolate vertically
    for y in range(rows):
        for x in range(cols):
            if y%space != 0:
                lastY = space * (y // space)
                nextY = lastY + space
                fracY = (y - lastY) / float(space)
                u_list[y,x,:] = lerpv(fracY, u_list[lastY, x, :], u_list[nextY, x, :])

    u_grid = u_list.reshape(rows * cols, dim)

    return u_grid

#print(create_mine_grid(rows=16, cols=16, dim=100, space=1, anchors=None, spherical=True, gaussian=True))