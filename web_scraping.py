import re
import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL'leri dosyadan oku
df = pd.read_csv('new134_freelancer_urls.csv')
freelancer_data = []

for index, row in df.iterrows():
    profile_url = row['Profile URL']
    print(f"\n[{index+1}/{len(df)}] İşleniyor: {profile_url}")  # ADIMI EKRNA BAS

    try:
        response = requests.get(profile_url, timeout=10)  # timeout koyduk, takılıp kalmasın
    except Exception as e:
        print(f"Hata oluştu: {e}")
        response = None

    if response and response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        try:
            amount_earned = soup.find('p', class_='EarningsText')
            completed_jobs_percentage = soup.find('fl-completed-jobs')
            if completed_jobs_percentage:
                percentage_text = completed_jobs_percentage.find('p', class_='text-small').get_text(strip=True)
            
            #usd_per_hour = soup.find('div', class_='NativeElement ng-star-inserted')
            reputation_elements = soup.select('fl-text.ReputationItemAmount div.NativeElement.ng-star-inserted')
            on_time = reputation_elements[0].get_text(strip=True) if len(reputation_elements) > 0 else 'N/A'
            on_budget = reputation_elements[1].get_text(strip=True) if len(reputation_elements) > 1 else 'N/A'
            accept_rate = reputation_elements[2].get_text(strip=True) if len(reputation_elements) > 2 else 'N/A'
            repeat_hire_rate = reputation_elements[3].get_text(strip=True) if len(reputation_elements) > 3 else 'N/A'

            #on_time = soup.select_one('fl-text.ReputationItemAmount div.NativeElement.ng-star-inserted')
            #on_budget = soup.select_one('fl-text.ReputationItemAmount div.NativeElement.ng-star-inserted')
            #repeat_hire_rate = soup.find('div', class_='repeat-hire-class')
            #accept_rate = soup.find('div', class_='accept-rate-class')
            country_time_text = soup.find('fl-col', class_='SupplementaryInfo')
            
            #continent = soup.find('div', class_='continent-class')
            reviews_count = soup.select_one('fl-review-count div.NativeElement.ng-star-inserted')
            overall_rating = soup.select_one('fl-rating div.ValueBlock.ng-star-inserted')
            
            education_values = soup.select('app-user-profile-resume-section div.ng-star-inserted')
            education = 1 if education_values else 0
            experience_section = soup.select_one('app-user-profile-experiences-redesign')
            if experience_section:
                work_experience_values = experience_section.select('fl-experience')
                
                work_experience = len(work_experience_values)
            else:
                work_experience = 0
            

            sertification_section = soup.select_one('app-user-profile-exams-grid-view')
            if sertification_section:
                sertification_values = sertification_section.select('fl-callout')
                
                number_of_certificates = len(sertification_values)
            else:
                number_of_certificates = 0
            
            refuse_rate = soup.find('div', class_='refuse-rate-class')
            gender = soup.find('div', class_='gender-class')

            usd_elements = soup.select('div.NativeElement.ng-star-inserted')

            # İçinde "USD" geçen div'i bul (ilkini al)
            usd_per_hour = next((el for el in usd_elements if 'USD' in el.get_text()), None)
           
                        
                    

            scraped_row = {
                'Username': row['Username'],
                'Profile URL': profile_url,
                'Amount Earned': amount_earned.text.strip() if amount_earned else 'N/A',
                'Completed Jobs %': completed_jobs_percentage.text.strip() if completed_jobs_percentage else 'N/A',
                'USD/Hour': usd_per_hour.text.strip() if usd_per_hour else 'N/A',
                'On Time': on_time,
                'On Budget': on_budget,
                'Repeat Hire Rate': repeat_hire_rate,
                'Accept Rate': accept_rate,
                'Country': country_time_text.text.strip() if country_time_text else 'N/A',
                
                'Reviews Count': reviews_count.text.strip() if reviews_count else 'N/A',
                'Overall Rating': overall_rating.text.strip() if overall_rating else 'N/A',
                'Education': education if education else 0,
                'Work Experience': work_experience if work_experience else 0,
                'Number of Certificates': number_of_certificates if number_of_certificates else 0,
                'Refuse Rate': refuse_rate.text.strip() if refuse_rate else 'N/A',
                'Gender': gender.text.strip() if gender else 'N/A'
            }

            freelancer_data.append(scraped_row)

            # ÇEKİLEN VERİLERİ EKRANA BAS
            print(f"Çekilen veriler: {scraped_row}")

        except Exception as e:
            print(f"Veri çekilirken hata oluştu: {e}")
    else:
        print(f"Sayfaya ulaşılamadı: {profile_url}")
        freelancer_data.append({
            'Username': row['Username'],
            'Profile URL': profile_url,
            'Amount Earned': 'N/A',
            'Completed Jobs %': 'N/A',
            'USD/Hour': 'N/A',
            'On Time': 'N/A',
            'On Budget': 'N/A',
            'Repeat Hire Rate': 'N/A',
            'Accept Rate': 'N/A',
            'Country': 'N/A',
            'Continent': 'N/A',
            'Reviews Count': 'N/A',
            'Overall Rating': 'N/A',
            'Education': 'N/A',
            'Work Experience': 'N/A',
            'Number of Certificates': 'N/A',
            'Refuse Rate': 'N/A',
            'Gender': 'N/A'
        })

# Veriyi kaydet
df_scraped = pd.DataFrame(freelancer_data)
df_scraped.to_csv('android_developer.csv', index=False)
print("\nWeb scraping tamamlandı ve veriler kaydedildi!")
