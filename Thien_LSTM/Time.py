import numpy as np
from geopy.distance import geodesic
from math import sqrt

turning_point_flow = 1500
speed_limit = 60
coef_a = -1.4648375
coef_b = 93.75

def calculate_time(scat_a, scat_b, flow_at_b):
    speed1 = min(speed_limit, (-coef_b - sqrt(coef_b**2 - 4 * coef_a * flow_at_b)) / (2 * coef_a))
    speed2 = min(speed_limit, (-coef_b + sqrt(coef_b**2 - 4 * coef_a * flow_at_b)) / (2 * coef_a))
    exact_speed = max(speed1, speed2)

    distance = geodesic(scat_a, scat_b).km 
    return distance / exact_speed * 60  # in minutes
