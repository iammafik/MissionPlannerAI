import ollama
import datetime
import csv

def create_schedule(prompt):
    response = ollama.chat(
        model="llama3",  # Ollama'da yüklü modelin adı
        messages=[
            {"role": "system", "content": "Sen bir kişisel görev yöneticisi ve planlayıcısın. Kullanıcının verdiği bilgiye göre günlük zaman çizelgesi öner."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['message']['content']


if __name__ == "__main__":
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    user_prompt = f"Yarın ({tomorrow.strftime('%d %B %Y')}) saat 9:00, 12:30 ve 15:00’te toplantılarım var. Günümü nasıl planlamalıyım?"

    print("📅 Günlük Plan:\n")
    plan = create_schedule(user_prompt)
    print(plan)


    def save_to_csv(plan_str):
        lines = plan_str.strip().split('\n')
        with open("plan.csv", "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Zaman", "Görev"])
            for line in lines:
                if ' - ' in line:
                    saat, görev = line.split(':', 1) if ':' in line else (line, '')
                    writer.writerow([saat.strip(), görev.strip()])


    save_to_csv(plan)
