import numpy as np
from geopy.distance import geodesic
from math import sqrt


# exact is at 1500.0064000273069, but this is good enough
turning_point_flow = 1500
speed_limit = 60
coef_a = -1.4648375
coef_b = 93.75

# coefficients = [coef_a, coef_b, 0]
# roots = np.roots(coefficients)
# flow_maximum_speed = (roots[0] + roots[1]) / 2
# max_capcity = coef_a * flow_maximum_speed * flow_maximum_speed + coef_b * flow_maximum_speed
# print(roots)
# print(flow_maximum_speed)
# print(max_capcity)

# scat_a and scat_b are in (latitude, longitude) tuples
def calculate_time(scat_a, scat_b, flow_at_b):
    speed = 0

    # naive approach, assume largest non-negative, because speed is king :) 
    # TODO: make this make more sense later (not gonna happen)
    speed1 = min(speed_limit, (-coef_b - sqrt(coef_b**2 - 4 * (coef_a) * flow_at_b)) / (2 * coef_a))
    speed2 = min(speed_limit, (-coef_b + sqrt(coef_b**2 - 4 * (coef_a) * flow_at_b)) / (2 * coef_a))
    exact_speed = max(speed1, speed2)

    distance = geodesic(scat_a, scat_b).km 
    
    # time = distance / speed, returns time in minutes
    return distance / exact_speed * 60
