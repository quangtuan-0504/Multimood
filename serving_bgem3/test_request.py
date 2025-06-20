import requests
import json

url = "http://0.0.0.0:8000/compute-score/"

payload = {
    "sentence_1": "The cat is on the mat",
    "sentence_2": "The cat sits on the mat"
}

headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    if response.status_code == 200:
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