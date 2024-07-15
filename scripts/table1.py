import pathlib
import re
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

patients = pd.read_csv("~/Desktop/meta.csv")
patients = patients[patients.Diagnosis.isin(["Healthy Control", "Dementia-AD senile onset"])]
patients = patients["AnonymizedID YearsEd Race PatientGender Ethnicity Handed DateOfBirth".split()]
race = patients.groupby("Race Ethnicity".split()).count().iloc[:, 0]
gender = patients.groupby("PatientGender".split()).count().iloc[:, 0]
handed = patients.groupby("Handed".split()).count().iloc[:, 0]
# screw it, hand-fix this column
patients.DateOfBirth = ['1959-01-01', '1991-02-07', '1968-08-08', '1941-12-21', '1988-12-29', '1952-09-27', '1939-12-26', '1933-02-28', '1949-03-15', '1962-01-24', '1932-01-01', '1945-05-20', '1946-10-30', '1970-03-12', '1941-12-21', '1989-07-02', '1989-07-02', '1939-12-26', '1961-08-25', '1952-09-27', '1934-03-04', '1942-11-21', '1933-01-24', '1989-07-02', '1965-04-23', '1944-03-21', '1989-07-02', '1942-07-14', '1965-03-26', '1989-07-02', '1970-03-12', '1932-01-01', '1961-03-02', '1946-10-30', '1965-04-23', '1988-12-29', '1988-12-29', '1942-07-14', '1956-10-08', '1934-03-04', '1944-03-21', '1975-06-06', '1989-03-20', '1931-01-27', '1939-03-03', '1965-03-26', '1931-01-27', '1989-07-02', '1988-12-29', '1989-03-20', '1944-09-21', '1942-11-21', '1939-03-03', '1938-10-10', '1961-09-22', '1934-11-29', '1989-03-20', '1944-09-21', '1989-03-20', '1962-01-24', '1975-06-06', '1968-08-08', '1953-11-13', '1961-09-13', '1980-04-18', '1947-08-11', '1947-09-24', '1934-11-29', '1933-02-28', '1959-01-01', '1934-11-29', '1946-10-30', '1953-11-13', '1945-05-20', '1951-03-09', '1947-09-24', '1971-01-20', '1971-01-20', '1939-12-26', '1989-07-02', '1933-02-28', '1959-05-29', '1961-03-02']
patients.DateOfBirth = pd.to_datetime(patients.DateOfBirth)

parent = pathlib.Path("/Users/abe/Desktop/dsst/Desktop/SSK/")
def get_test_date(id):
  for g in parent.glob(f"*{id}*"):
    date = re.findall("20\d\d-\d\d-\d\d", g.stem)
    date = pd.to_datetime(date[0])
    return date
patients["testdate"] = [get_test_date(id) for id in patients.AnonymizedID]
patients["age"] = (patients.testdate - patients.DateOfBirth).dt.days // 365
plt.hist(patients.age)
patients.age = np.select([patients.age < 65, patients.age < 75, patients.age < 85, patients.age >= 85], ['< 65', '65-74', '75-84', '85+'])
age = patients.groupby("age").count().iloc[:, 0]

print(pd.concat([age, gender, handed, race]))