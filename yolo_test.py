from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("traffic_images/traffic1.jpg")

car_count = 0
bus_count = 0
truck_count = 0

for box in results[0].boxes:

    cls = int(box.cls[0])

    class_name = model.names[cls]

    if class_name == "car":
        car_count += 1

    elif class_name == "bus":
        bus_count += 1

    elif class_name == "truck":
        truck_count += 1

print("Cars:", car_count)
print("Buses:", bus_count)
print("Trucks:", truck_count)

total = car_count + bus_count + truck_count

print("Total Vehicles:", total)