import math

def compute_lateral_error(left_x, right_x, frame_width, hfov_deg):
    runway_width = right_x - left_x
    if runway_width == 0:
        return 0.0
    centerline_x = (left_x + right_x) / 2.0 # average of xL and xR
    image_center_x = frame_width / 2.0
    dx_pixels = image_center_x - centerline_x

    # lateral error in runway widths
    # +1 = 1 runway-width right, -1 = left
    lateral_error_rw = dx_pixels / runway_width

    hfov_rad = math.radians(hfov_deg)
    # compute focal length in pixels from hfov
    fx = (frame_width / 2.0) / math.tan(hfov_rad / 2.0)

    lat_err_rad = math.atan(dx_pixels / fx)
    lat_err_deg = math.degrees(lat_err_rad)

    return lateral_error_rw, lat_err_deg

def compute_vertical_error(base_y, frame_height, ideal_base_y_frac, vfov_deg):
    ideal_base_y = frame_height * ideal_base_y_frac

    vert_err_px = base_y - ideal_base_y

    # use linear estimate (TODO: or use focal length?)
    deg_per_px = vfov_deg / frame_height
    vert_err_deg = deg_per_px * vert_err_px

    return vert_err_px, vert_err_deg