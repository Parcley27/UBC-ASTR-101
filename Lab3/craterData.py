import csv
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

# Shared loading for Lab 3: reads the raw measurements straight out of the group spreadsheet.
# Only the measured input columns are read (by header name); the spreadsheet's computed columns are ignored.

spreadsheetPath = "Cratering.xlsx"
csvPath = "cratering.csv"

gravity = 980.0 # cm/s^2, lab manual value (the group spreadsheet used 981)

# Drop-height convention (unconfirmed): drop height is measured from the floor of the box,
# so the ball falls the drop height minus the sand depth. Change it here only.
def fallHeightFrom(dropHeight, sandDepth):
    return dropHeight - sandDepth # cm

# spreadsheet header -> our field name
inputColumns = {
    "Impact number": "impact",
    "Sand depth (cm)": "sandDepth",
    "Impactor material": "material",
    "Commonality": "commonality",
    "Drop height (cm)": "dropHeight",
    "Impactor diameter (cm)": "ballDiameter",
    "Impactor mass (g)": "mass",
    "Crater diameter (mm)": "craterDiameter",
}

# "Commonality" records what was held fixed, which identifies the set
setByCommonality = {
    "Density, Height": "mass",
    "Diameter, Height": "density",
    "Density, Diameter": "velocity",
}

mainNamespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def columnLetters(cellReference):
    return "".join(character for character in cellReference if character.isalpha())


def readSheet(path):
    """Return the first worksheet as a list of {column letter: cell value} rows (cached values, no formulas)."""
    tag = lambda name: f"{{{mainNamespace}}}{name}"

    with zipfile.ZipFile(path) as workbook:
        sharedStrings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            stringTable = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in stringTable.findall(tag("si")):
                sharedStrings.append("".join(text.text or "" for text in item.iter(tag("t"))))

        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))

    sheetData = sheet.find(tag("sheetData"))
    if sheetData is None:
        return []

    rows = []
    for row in sheetData:
        values = {}
        for cell in row:
            valueNode = cell.find(tag("v"))
            if valueNode is None or valueNode.text is None:
                continue

            cellType = cell.get("t")
            if cellType == "s":
                value = sharedStrings[int(valueNode.text)]
            elif cellType in ("str", "inlineStr", "b", "e"):
                value = valueNode.text
            else:
                value = float(valueNode.text)

            values[columnLetters(cell.get("r", ""))] = value
        rows.append(values)

    return rows


def loadImpacts(path = spreadsheetPath):
    """Read the raw measurements into a dict of numpy arrays, one entry per impact."""
    rows = readSheet(path)

    header = rows[0]
    letterFor = {}
    for letter, name in header.items():
        if name in inputColumns:
            letterFor[inputColumns[name]] = letter

    missing = set(inputColumns.values()) - set(letterFor)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    records = []
    for row in rows[1:]:
        impactValue = row.get(letterFor["impact"])
        if not isinstance(impactValue, float):
            continue # blank spacer rows and the desmos link

        record = {field: row.get(letter) for field, letter in letterFor.items()}
        record["set"] = setByCommonality[record["commonality"]]
        records.append(record)

    records.sort(key = lambda record: record["impact"])

    data = {
        "impact": np.array([int(record["impact"]) for record in records]),
        "set": np.array([record["set"] for record in records]),
        "material": np.array([record["material"] for record in records]),
    }
    for field in ("sandDepth", "dropHeight", "ballDiameter", "mass", "craterDiameter"):
        data[field] = np.array([float(record[field]) for record in records])

    data["fallHeight"] = fallHeightFrom(data["dropHeight"], data["sandDepth"]) # cm

    return data


def writeCsv(data, path = csvPath):
    """Write the raw measurements (as read from the spreadsheet) to a plain CSV."""
    header = ["impact", "set", "material", "sand depth (cm)", "drop height (cm)", "ball diameter (cm)", "mass (g)", "crater diameter (mm)"]

    with open(path, "w", newline = "", encoding = "utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        for index in range(len(data["impact"])):
            writer.writerow([
                data["impact"][index],
                data["set"][index],
                data["material"][index],
                f"{data['sandDepth'][index]:g}",
                f"{data['dropHeight'][index]:g}",
                f"{data['ballDiameter'][index]:g}",
                f"{data['mass'][index]:g}",
                f"{data['craterDiameter'][index]:g}",
            ])
