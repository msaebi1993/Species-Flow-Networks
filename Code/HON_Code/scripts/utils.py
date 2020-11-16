import csv


def GetPortData(fn, field, delim):
    ports = {}
    with open(fn) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delim)
        for row in reader:
            ports[row[field]] = row
    return ports