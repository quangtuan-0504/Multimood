import requests
import json

# URL của endpoint FastAPI
url = "http://0.0.0.0:8000/compute-score/"

# Dữ liệu đầu vào: hai câu để so sánh
payload = {
    "sentence_1": "The cat is on the mat",
    "sentence_2": "The cat sits on the mat"
}

# Tiêu đề yêu cầu
headers = {
    "Content-Type": "application/json"
}

try:
    # Gửi yêu cầu POST
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    # Kiểm tra trạng thái phản hồi
    if response.status_code == 200:
        # In kết quả
        result = response.json()
        print("Kết quả từ server:")
        print(f"Câu 1: {result['sentence_1']}")
        print(f"Câu 2: {result['sentence_2']}")
        print(f"Điểm BGE: {result['bge_score']}")
    else:
        print(f"Lỗi: {response.status_code} - {response.text}")

except requests.exceptions.ConnectionError:
    print("Không thể kết nối đến server. Vui lòng kiểm tra xem server đã chạy chưa.")
except Exception as e:
    print(f"Đã xảy ra lỗi: {str(e)}")