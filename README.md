# Predicția Soldului Energetic din Sistemul Energetic Național (SEN) pentru luna decembrie 2024

## Prezentarea Generală

Acest proiect are ca scop predicția soldului total (diferența dintre producție și consumul de energie electrică) în Sistemul Energetic Național (SEN) al României pentru luna decembrie 2024. Scopul este de a implementa și evalua două modele de învățare automată pentru regresie: arborele de decizie ID3 și clasificarea Bayesiană.

## Scopul Proiectului

Obiectivul principal este de a utiliza tehnici de învățare automată pentru a prezice soldul energetic, care este calculat ca diferența dintre producția și consumul de energie electrică din SEN. Modelele implementate sunt:
- **ID3**: un algoritm bazat pe arbori de decizie, adaptat pentru regresie.
- **Clasificare Bayesiană**: o metodă probabilistică folosită pentru predicțiile categorice ale soldului.

Aceste modele sunt evaluate pe baza unor metrici de performanță precum RMSE (eroarea pătratică medie), MAE (eroarea absolută medie) și acuratețea.

## Datele Folosite

Datele utilizate provin de pe platforma Transelectrica SEN Grafic și includ informații despre consumul și producția de energie electrică din SEN. Setul de date conține următoarele coloane:
- **Data**: Timpul specific al înregistrării.
- **Consum[MW]**: Consumul total de energie electrică.
- **Producție[MW]**: Producția totală de energie.
- Diverse surse de producție, inclusiv: **Carbune, Hidrocarburi, Ape, Nuclear, Eolian, Foto, Biomasă**.
- **Sold[MW]**: Diferența dintre producție și consum.

Datele pentru luna decembrie 2024 au fost excluse din datele de antrenare și utilizate exclusiv pentru testare. În plus, au fost adăugate caracteristici suplimentare, cum ar fi suma surselor de energie intermitente (eolian și solar) și surselor constante (nuclear, cărbune și hidrocarburi).

## Pași pentru Rularea Codului

1. **Clonarea Repozitoriului**
    Clonați acest proiect de pe GitHub:
   
    `git clone https://github.com/LauraC360/AP1_ML.git`

2. **Instalarea Dependențelor**
Instalați pachetele necesare utilizând `pip`. De exemplu:

    `pip install scikit-learn`

3. **Preprocesarea Datelor**
- Scriptul `aggregate_train_data.py` prelucrează datele pentru anii 2022 și 2023 și le salvează într-un fișier CSV (`daily_data.csv`).
- Scriptul `aggregate_test_data.py` prelucrează datele pentru luna decembrie 2024 și le salvează într-un fișier CSV (`daily_december_2024.csv`).

4. **Antrenarea Modelului**
- Pentru antrenarea modelului ID3, rulați scriptul `main.py`. Acesta va aplica modelul de regresie ID3 și va salva rezultatele într-un fișier CSV cu predicțiile pentru luna decembrie 2024.
- La fel, puteți încerca să rulați și oricare dintre fișierele `id3.py`, `id3_2.py`, `bayes.py` pentru a avea mai multe predicții și rezultate utile de performanță

5. **Evaluarea Performanței**
După rularea scriptului, veți obține un fișier CSV (`comparison_results.csv`) cu rezultatele predicțiilor și evaluarea performanței bazată pe metricele RMSE, MAE și acuratețe.

## Concluzii

Proiectul a demonstrat aplicabilitatea algoritmilor de învățare automată pentru predicția soldului energetic, arătând o performanță mai bună cu algoritmul ID3 în comparație cu clasificarea Bayesiană. Aceste predicții pot fi utilizate pentru a înțelege mai bine dinamica sistemului energetic din România și pentru a optimiza gestionarea resurselor.

    




