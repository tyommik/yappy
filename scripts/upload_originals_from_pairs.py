import os
import re
import requests
from pathlib import Path

def upload_videos():
    directory = Path('../output')
    pattern = re.compile(r'^(\d+)_0_([a-f0-9-]+)\.mp4$')
    
    for file in directory.iterdir():
        if file.is_file():
            match = pattern.match(file.name)
            if match:
                idx, uuid = match.groups()
                file_url = f"http://127.0.0.1:8000/output/{file.name}"
                
                payload = {
                    "link": file_url
                }
                
                response = requests.post(
                    "http://192.168.204.212:8000/add-video",
                    json=payload,
                    headers={
                        'Content-Type': 'application/json',
                        'User-Agent': 'Python/3.x'
                    }
                )
                
                if response.status_code == 200:
                    print(f"Успешно загружено видео: {file.name}")
                else:
                    print(f"Ошибка при загрузке видео {file.name}: {response.status_code}")

if __name__ == "__main__":
    upload_videos()
