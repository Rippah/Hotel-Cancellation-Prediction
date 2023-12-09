import pandas as pd
import numpy as np

# Učitaj podatke iz CSV datoteke
data = pd.read_csv('booking.csv')

#############################################################################################################################################
# Pregled procenta nedostajućih vrednosti za svaku kolonu
for column in data.columns:
    missing_percentage = (data[column].isnull().sum() / len(data)) * 100
    print(f"{column}: {missing_percentage:.2f}% nedostajućih vrednosti.")
#############################################################################################################################################

#############################################################################################################################################
# Pretvori 'date of reservation' u datetime format uz odgovarajući format datuma
data['date of reservation'] = pd.to_datetime(data['date of reservation'], format='%m/%d/%Y', errors='coerce')

# Dodaj kolone 'year', 'month' i 'year_month' na osnovu datuma rezervacije
data['year'] = data['date of reservation'].dt.year
data['month'] = data['date of reservation'].dt.month
data['year_month'] = data['date of reservation'].dt.to_period('M')  # Period sa mesecem

# Izdvoji rezervacije koje su otkazane
otkazane_rezervacije = data[data['booking status'] == 'Canceled']

# Izračunaj procentualni udeo otkazanih rezervacija po godinama i mesecima
procenti_otkazanih_godina_meseci = otkazane_rezervacije.groupby(['year', 'month']).size() / data.groupby(['year', 'month']).size() * 100

# Ispisi rezultate
print("Procenat otkazanosti po godinama i mesecima:")
print(procenti_otkazanih_godina_meseci)

# Broj rezervacija za određeni mesec i godinu (na primer, za 2018. godinu i maj mesec)
print(data[(data['year'] == 2018) & (data['month'] == 5)].shape[0])
data.drop(['year', 'month', 'year_month'], inplace=True, axis=1)
#############################################################################################################################################

#############################################################################################################################################
# Izračunaj procentualni udeo otkazanih rezervacija po broju dece
procenti_otkazanih_children = otkazane_rezervacije.groupby('number of children').size() / data.groupby('number of children').size() * 100

# Ispisi rezultate
print("Procenat otkazanosti po broju dece:")
print(procenti_otkazanih_children)

# Izračunaj procentualni udeo otkazanih rezervacija po broju odraslih
procenti_otkazanih_adults = otkazane_rezervacije.groupby('number of adults').size() / data.groupby('number of adults').size() * 100

# Ispisi rezultate
print("Procenat otkazanosti po broju odraslih:")
print(procenti_otkazanih_adults)
#############################################################################################################################################

#############################################################################################################################################
# Izračunavanje IQR za svako obeležje
Q1 = data['average price'].quantile(0.25)
Q3 = data['average price'].quantile(0.75)
IQR = Q3 - Q1

# Identifikacija autlajera
outliers = (data['average price'] < (Q1 - 1.5 * IQR)) | (data['average price'] > (Q3 + 1.5 * IQR))

print(data[outliers].shape[0])
#############################################################################################################################################

#############################################################################################################################################
from scipy.stats import f_oneway, chi2_contingency

new_data = data.drop(['Booking_ID', 'P-not-C'], axis=1)

# Izaberi numeričko obeležje
numericko_oznacavanje = 'average price'

# Inicijalizuj prazne liste za rezultate
rezultati_anova = []
rezultati_chi2 = []

# Iteriraj kroz sve kategoričke kolone osim 'average price'
for kategoricko_oznacavanje in new_data.select_dtypes(include=['object']).columns.difference([numericko_oznacavanje, 'Booking_ID', 'P-not-C']):
    # Grupiši podatke po kategoričkom obeležju i izračunaj ANOVA
    grupisani_podaci = [new_data[numericko_oznacavanje][new_data[kategoricko_oznacavanje] == kategorija] for kategorija in new_data[kategoricko_oznacavanje].unique()]

    # Provera da li postoje podaci za svaku grupu pre nego što izračunamo ANOVA
    if all(len(grupa) > 0 for grupa in grupisani_podaci):
        rezultat_anova = f_oneway(*grupisani_podaci)

        # Dodaj rezultate ANOVA u listu
        rezultati_anova.append({
            'Kategoricko Oznacavanje': kategoricko_oznacavanje,
            'F-statistika': rezultat_anova.statistic,
            'P-vrednost': rezultat_anova.pvalue
        })

        # Kreiraj kontingencijsku tablicu za chi-square test
        contingency_table = pd.crosstab(new_data[numericko_oznacavanje], new_data[kategoricko_oznacavanje])

        # Izračunaj koeficijent kontingencije
        chi2, p, _, _ = chi2_contingency(contingency_table)

        # Dodaj rezultate chi-square testa u listu
        rezultati_chi2.append({
            'Kategoricko Oznacavanje': kategoricko_oznacavanje,
            'Koeficijent Kontingencije': chi2,
            'P-vrednost': p
        })

# Prikazi rezultate ANOVA
print("Rezultati ANOVA:")
for rezultat in rezultati_anova:
    print(f"\nKategoricko Oznacavanje: {rezultat['Kategoricko Oznacavanje']}")
    print(f"F-statistika: {rezultat['F-statistika']}")
    print(f"P-vrednost: {rezultat['P-vrednost']}")

# Prikazi rezultate chi-square testa
print("\nRezultati Chi-square testa:")
for rezultat in rezultati_chi2:
    print(f"\nKategoricko Oznacavanje: {rezultat['Kategoricko Oznacavanje']}")
    print(f"Koeficijent Kontingencije: {rezultat['Koeficijent Kontingencije']}")
    print(f"P-vrednost: {rezultat['P-vrednost']}")

# Rezultati ANOVA
min_pvalue_anova = min(rezultati_anova, key=lambda x: x['P-vrednost'])
print(f"Najmanja P-vrednost u rezultatima ANOVA: {min_pvalue_anova['P-vrednost']}")
print(f"Odgovarajuće kategoricko obeležje u ANOVA: {min_pvalue_anova['Kategoricko Oznacavanje']}")

# Rezultati Chi-square testa
min_pvalue_chi2 = min(rezultati_chi2, key=lambda x: x['P-vrednost'])
print(f"Najmanja P-vrednost u rezultatima Chi-square testa: {min_pvalue_chi2['P-vrednost']}")
print(f"Odgovarajuće kategoricko obeležje u Chi-square testu: {min_pvalue_chi2['Kategoricko Oznacavanje']}")

    

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

kategoricka_obelezja = new_data.columns.difference([numericko_oznacavanje])

for i, obelezje1 in enumerate(kategoricka_obelezja):
    for j in range(i+1, len(kategoricka_obelezja)):
        obelezje2 = kategoricka_obelezja[j]
        v = cramers_v(new_data[obelezje1], new_data[obelezje2])
        if v > 0.7:
          print(f"Par obeležja: ({obelezje1}, {obelezje2}), Cramerov V koeficijent: {v}")

#############################################################################################################################################

#############################################################################################################################################
import matplotlib.pyplot as plt
import seaborn as sns

# Podesite paletu boja za svaku klasu
trenutna_klasa = 'number of children'
palette = sns.color_palette("husl", n_colors=len(data[trenutna_klasa].unique()))

# Iterirajte kroz obeležja i iscrtajte histograme po klasama
features = new_data.columns.difference(['average price'])

for feature in features:
    plt.figure(figsize=(8, 6))
    for klasa, color in zip(data[trenutna_klasa].unique(), palette):
        sns.histplot(data[data[trenutna_klasa] == klasa][feature], kde=True, color=color, label=f'Klasa {klasa}')

    plt.title(f'Histogram za Obeležje: {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frekvencija')
    plt.legend()
    plt.show()

#############################################################################################################################################