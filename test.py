import csv

# Dữ liệu cần lưu
data = [
    {"Tên": "Hải", "Tuổi": 25, "Thành phố": "Hà Nội"},
    {"Tên": "Lan", "Tuổi": 30, "Thành phố": "Hồ Chí Minh"},
    {"Tên": "Minh", "Tuổi": 22, "Thành phố": "Đà Nẵng"}
]

# Tên file CSV
file_name = "data_dict.csv"

# Ghi dữ liệu vào file CSV
with open(file_name, mode="w", encoding="utf-8", newline="") as file:
    fieldnames = ["Tên", "Tuổi", "Thành phố"]  # Đặt các cột
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()  # Ghi dòng tiêu đề
    writer.writerows(data)  # Ghi dữ liệu

print(f"Dữ liệu đã được lưu vào file {file_name}")
