"""Inference definitions extracted from public_packet_camera_timing.py; no training entry point."""
def camera_receipt(controller_stamp_s, camera_stamps_ns, previous_stamps=None):
    stamps = tuple((int(value) for value in camera_stamps_ns))
    controller = float(controller_stamp_s)
    latest = max(controller, max(stamps) * 1e-09)
    camera_ages = [latest - stamp * 1e-09 for stamp in stamps]
    state_age = latest - controller
    distinct = previous_stamps is None or all((a > b for a, b in zip(stamps, previous_stamps)))
    accepted = bool(len(stamps) == 3 and min(stamps) > 0 and distinct and (0.0 <= state_age <= 0.010000001) and (min(camera_ages) >= -1e-09) and (max(camera_ages) <= 0.010000001) and (max(stamps) - min(stamps) <= 10000000))
    return dict(accepted=accepted, available_packet_time_s=latest, controller_age_s=state_age, camera_ages_s=camera_ages, distinct_images=bool(distinct), maximum_state_age_s=0.01, sensor_data_already_delivered=True)
