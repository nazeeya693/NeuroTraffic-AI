from vehicle_counter import count_vehicles

cars, buses, trucks, total = count_vehicles(
    "traffic_images/traffic1.jpg"
)

print("Cars:", cars)
print("Buses:", buses)
print("Trucks:", trucks)
print("Total:", total)