import pandas as pd
from uuid import uuid4

orig = pd.read_csv("C:\\Users\\nadir\\FinalProject\\LLGC-main\\fakes.csv")

new_rows = [
    {
        "id": uuid4().hex[:24],
        "title": "advanced cake frosting",
        "authors.name": "['Random Baker']",
        "year": 2021,
        "fos.name": "Cooking",
        "n_citation": 0,
        "references": "[]",
        "abstract": "This paper explores advanced frosting techniques for cakes."
    },
    {
        "id": uuid4().hex[:24],
        "title": "introduction to neural widgets",
        "authors.name": "['Nadir Yaakov']",
        "year": 2022,
        "fos.name": "Computer Science",
        "n_citation": 2,
        "references": "[]",
        "abstract": "Neural widgets are a fictional concept used for testing datasets."
    },
    {
        "id": uuid4().hex[:24],
        "title": "basic medical tools",
        "authors.name": "['Ben Zakai']",
        "year": 2019,
        "fos.name": "Medical",
        "n_citation": 1,
        "references": "[]",
        "abstract": "An overview of commonly used basic medical instruments."
    },
    {
        "id": uuid4().hex[:24],
        "title": "pc cooling techniques",
        "authors.name": "['Nadir Yaakov']",
        "year": 2021,
        "fos.name": "Engineering",
        "n_citation": 3,
        "references": "[]",
        "abstract": "Discussion of air and liquid cooling methods for personal computers."
    },
    {
        "id": uuid4().hex[:24],
        "title": "food chemistry basics",
        "authors.name": "['Random']",
        "year": 2018,
        "fos.name": "Chemistry",
        "n_citation": 0,
        "references": "[]",
        "abstract": "This paper introduces chemical processes involved in cooking."
    },
]

df_new = pd.concat([orig, pd.DataFrame(new_rows)], ignore_index=True)
out_path = "C:\\Users\\nadir\\FinalProject\\LLGC-main\\fakes.csv"
df_new.to_csv(out_path, index=False)

out_path, len(df_new)
