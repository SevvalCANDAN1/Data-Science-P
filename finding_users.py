import requests
import pandas as pd
import time

# Freelancer API token
ACCESS_TOKEN = ''
headers = {'Authorization': f'Bearer {ACCESS_TOKEN}'}

# Kullanıcıları API ile al
search_url = 'https://www.freelancer.com/api/users/0.1/users/directory/'
search_params = {'query': 'android developer', 'limit': 250}  

search_response = requests.get(search_url, headers=headers, params=search_params)

# Hata durumunda yeniden denemek için
if search_response.status_code == 200:
    search_data = search_response.json()
    freelancers = search_data['result']['users']
    
    # URL oluştur ve sakla
    freelancer_urls = []

    for freelancer in freelancers:
        username = freelancer['username']
        profile_url = f'https://www.freelancer.com/u/{username}'
        freelancer_urls.append({'Username': username, 'Profile URL': profile_url})
    
    # Veriyi CSV dosyasına kaydedelim
    df = pd.DataFrame(freelancer_urls)
    df.to_csv('freelancer_urls.csv', index=False)
    print("Kullanıcı URL'leri dosyaya kaydedildi!")

else:
    print(f"Hata oldu: {search_response.status_code}")
    print(search_response.text)
    # Hata durumunda 10 saniye bekleyip tekrar deniyoruz
    time.sleep(10)
    search_response = requests.get(search_url, headers=headers, params=search_params)
