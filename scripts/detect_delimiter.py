import csv

path = "data/raw/form5500/f_5500_2024_latest.csv"

with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
    sample = f.read(20000)

dialect = csv.Sniffer().sniff(sample, delimiters=[",","|","\t",";"])
print("Detected delimiter:", repr(dialect.delimiter))

# show how many columns we get if we use it
first_line = sample.splitlines()[0]
print("First line length:", len(first_line))
print("Split columns:", len(first_line.split(dialect.delimiter)))