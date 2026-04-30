#!/usr/bin/env python3

import csv
import argparse
from pathlib import Path

ROWS = [
    {"Board":"FIU_SC2","ReceiveDetectEvent":"fiber cut","OutputBoard":"SC2","OutputType":"transmit downstream","Output":"OTS_LOS_A"},
    {"Board":"OA_FIU","ReceiveDetectEvent":"fiber cut","OutputBoard":"FIU","OutputType":"transmit downstream","Output":"OTS_LOS_B"},
    {"Board":"OM_OA","ReceiveDetectEvent":"fiber cut","OutputBoard":"OA","OutputType":"transmit downstream","Output":"OTS_LOS_C"},
    {"Board":"SC2_FIU","ReceiveDetectEvent":"fiber cut","OutputBoard":"FIU","OutputType":"transmit downstream","Output":"OTS_LOS_O"},
    {"Board":"FIU_OA","ReceiveDetectEvent":"fiber cut","OutputBoard":"OA","OutputType":"transmit downstream","Output":"OTS_LOS_O"},
    {"Board":"FIU_FIU","ReceiveDetectEvent":"fiber cut","OutputBoard":"FIU","OutputType":"transmit downstream","Output":"OTS_LOS_B"},
    {"Board":"OA_OD","ReceiveDetectEvent":"fiber cut","OutputBoard":"OD","OutputType":"transmit downstream","Output":"OMS_LOS_A"},
    {"Board":"OD_OM","ReceiveDetectEvent":"fiber cut","OutputBoard":"OM","OutputType":"transmit downstream","Output":"OMS_LOS_A"},
    {"Board":"Line_OM","ReceiveDetectEvent":"fiber cut","OutputBoard":"OM","OutputType":"transmit downstream","Output":"OMS_LOS_A"},
    {"Board":"OD_Line","ReceiveDetectEvent":"fiber cut","OutputBoard":"Line","OutputType":"transmit downstream","Output":"OCh_LOS_P"},

    {"Board":"OA","ReceiveDetectEvent":"board faulty","OutputBoard":"FIU;OA","OutputType":"transmit downstream;locally report","Output":"OTS_LOS_C;OTS_A_P"},
    {"Board":"FIU","ReceiveDetectEvent":"board faulty","OutputBoard":"FIU;FIU","OutputType":"transmit downstream;locally report","Output":"OTS_LOS_C;OTS_A_P"},
    {"Board":"SC2","ReceiveDetectEvent":"board faulty","OutputBoard":"FIU;SC2","OutputType":"transmit downstream;locally report","Output":"OTS_LOS_C;OTS_A_P"},
    {"Board":"OM","ReceiveDetectEvent":"board faulty","OutputBoard":"OD;OM","OutputType":"transmit downstream;locally report","Output":"OMS_LOS_A;OMS_A_P"},
    {"Board":"Line","ReceiveDetectEvent":"board faulty","OutputBoard":"Line;Line","OutputType":"transmit downstream;locally report","Output":"OCh_LOS_P;OCh_A_P"},
    {"Board":"XCON","ReceiveDetectEvent":"board faulty","OutputBoard":"XCON;XCON","OutputType":"transmit downstream;locally report","Output":"OCh_LOS_P;OCh_A_P"},
    {"Board":"Tributary","ReceiveDetectEvent":"board faulty","OutputBoard":"Tributary;Tributary","OutputType":"transmit downstream;locally report","Output":"OCh_LOS_P;OCh_A_P"},
    {"Board":"OD","ReceiveDetectEvent":"board faulty","OutputBoard":"Tributary;OD","OutputType":"transmit downstream;locally report","Output":"OCh_LOS_P;OMS_A_P"},

    {"Board":"OA","ReceiveDetectEvent":"lose input light","OutputBoard":"FIU;OA","OutputType":"transmit downstream;locally report","Output":"OTS_PMI;OTS_A"},

    {"Board":"FIU","ReceiveDetectEvent":"OTS_BDI_A","OutputBoard":"FIU;OM;OD","OutputType":"transmit upstream;transmit upstream;transmit downstream","Output":"OTS_BDI;OMS_BDI;OMS_SSF"},
    {"Board":"FIU","ReceiveDetectEvent":"OTS_LOS_B","OutputBoard":"FIU;OD","OutputType":"transmit downstream;transmit downstream","Output":"OTS_LOS_A;OMS_SSF_B"},
    {"Board":"FIU","ReceiveDetectEvent":"OTS_LOS_C","OutputBoard":"FIU;OM;OD;FIU","OutputType":"transmit upstream;transmit downstream;transmit downstream;transmit downstream","Output":"OTS_BDI_A;OMS_LOS_A;OMS_SSF_P;OTS_PMI"},
    {"Board":"FIU","ReceiveDetectEvent":"OTS_LOS_O","OutputBoard":"FIU;OM;OD","OutputType":"transmit upstream;transmit upstream;transmit downstream","Output":"OTS_BDI_A;OMS_BDI_O;OMS_SSF_O"},
    {"Board":"FIU","ReceiveDetectEvent":"OTS_LOS_P","OutputBoard":"FIU;OM;OD","OutputType":"transmit downstream;transmit downstream;transmit downstream","Output":"OTS_LOS_C;OMS_LOS_A;OMS_SSF_J"},

    {"Board":"OD","ReceiveDetectEvent":"OMS_SSF","OutputBoard":"Tributary","OutputType":"transmit downstream","Output":"OCh_LOS_P"},
    {"Board":"OD","ReceiveDetectEvent":"OMS_SSF_O","OutputBoard":"Tributary","OutputType":"transmit downstream","Output":"OCh_LOS_P"},
    {"Board":"OD","ReceiveDetectEvent":"OMS_SSF_A","OutputBoard":"Tributary","OutputType":"transmit downstream","Output":"OCh_LOS_P"},
    {"Board":"OD","ReceiveDetectEvent":"OMS_SSF_B","OutputBoard":"Tributary","OutputType":"transmit downstream","Output":"OCh_LOS_P"},
    {"Board":"OD","ReceiveDetectEvent":"OMS_SSF_C","OutputBoard":"Tributary","OutputType":"transmit downstream","Output":"OCh_LOS_P"},
    {"Board":"OD","ReceiveDetectEvent":"OMS_LOS_A","OutputBoard":"Tributary","OutputType":"transmit downstream","Output":"OCh_LOS_P"},
    {"Board":"OD","ReceiveDetectEvent":"OMS_SSF_E","OutputBoard":"Tributary","OutputType":"transmit downstream","Output":"OCh_LOS_P"},
    {"Board":"OD","ReceiveDetectEvent":"OMS_SSF_F","OutputBoard":"Tributary","OutputType":"transmit downstream","Output":"OCh_LOS_P"},
    {"Board":"OD","ReceiveDetectEvent":"OMS_SSF_P","OutputBoard":"Tributary","OutputType":"transmit downstream","Output":"OCh_LOS_P"},
    {"Board":"OD","ReceiveDetectEvent":"OMS_SSF_J","OutputBoard":"Tributary","OutputType":"transmit downstream","Output":"OCh_LOS_P"},
    {"Board":"OD","ReceiveDetectEvent":"OMS_SSF_K","OutputBoard":"Tributary","OutputType":"transmit downstream","Output":"OCh_LOS_P"},
    {"Board":"OD","ReceiveDetectEvent":"OMS_A_P","OutputBoard":"Tributary","OutputType":"transmit downstream","Output":"OCh_A_P"},

    {"Board":"OA","ReceiveDetectEvent":"OTS_LOS_O","OutputBoard":"FIU;OM;OD;OA","OutputType":"transmit downstream;transmit downstream;transmit downstream;transmit downstream","Output":"OTS_LOS_B;OMS_LOS_A;OMS_SSF_C;OTS_A_P"},
    {"Board":"OA","ReceiveDetectEvent":"OTS_LOS_C","OutputBoard":"FIU;OM;OD","OutputType":"transmit upstream;transmit downstream;transmit downstream","Output":"OTS_BDI_A;OMS_LOS_A;OMS_SSF_F"},
    {"Board":"OA","ReceiveDetectEvent":"OTS_A_P","OutputBoard":"OD","OutputType":"transmit downstream","Output":"OMS_A_P"},
    {"Board":"SC2","ReceiveDetectEvent":"OTS_LOS_A","OutputBoard":"FIU;OM;OD","OutputType":"transmit upstream;transmit downstream;transmit downstream","Output":"OTS_BDI_A;OTS_LOS_O;OMS_SSF_A"},
    {"Board":"OM","ReceiveDetectEvent":"OMS_LOS_A","OutputBoard":"OD","OutputType":"transmit downstream","Output":"OMS_SSF_E"},
]

FIELDS = ["Board","ReceiveDetectEvent","OutputBoard","OutputType","Output"]

def _read_extra_rules(extra_path: Path):
    extra_rows = []
    with extra_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filtered_row = {k: row[k] for k in FIELDS if k in row}
            extra_rows.append(filtered_row)
    return extra_rows

def write_rules_csv(out_path: Path, overwrite: bool=True, extra_rows=None):
    all_rows = list(ROWS)
    if extra_rows:
        all_rows.extend(extra_rows)
    seen = set()
    filtered_rows = []
    for row in all_rows:
        key = tuple(row.get(field, "") for field in FIELDS)
        if key in seen:
            continue
        if "OTU" in row.get("Board", "") or "OTU" in row.get("OutputBoard", ""):
            continue
        seen.add(key)
        filtered_rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists (use --force to overwrite)")
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in filtered_rows:
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str,
        default="outputs/Static files/rule_database_new.csv",
        help="Output CSV path")
    parser.add_argument("--force", action="store_true", help="Overwrite if file exists")
    parser.add_argument("--extra", type=str, default=None, help="Optional extra CSV file to merge")
    args = parser.parse_args()

    out = Path(args.out)
    extra_rows = None
    if args.extra:
        extra_path = Path(args.extra)
        extra_rows = _read_extra_rules(extra_path)

    write_rules_csv(out, overwrite=args.force or True, extra_rows=extra_rows)
    print(f"[ok] Wrote electrical-layer rules to: {out}")

    from collections import Counter
    c = Counter(r["Board"] for r in ROWS)
    print("[info] Row counts by Board:", dict(c))
    all_rows_to_check = list(ROWS)
    if extra_rows:
        all_rows_to_check.extend(extra_rows)
    assert all("OTU" not in r["Board"] and "OTU" not in r["OutputBoard"] for r in all_rows_to_check), "Found legacy OTU reference"

if __name__ == "__main__":
    main()