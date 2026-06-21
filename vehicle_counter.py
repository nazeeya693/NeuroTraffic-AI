from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def count_vehicles(image_path):

    results = model(image_path, save=True)

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

    total = car_count + bus_count + truck_count

    return car_count, bus_count, truck_count, total