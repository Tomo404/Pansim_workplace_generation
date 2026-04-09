from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from data_loader import load_counts_from_tensor_files
from impute_company_counts import impute_counts_to_national_targets


# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent

TENSOR_PATH = BASE_DIR / "ksh_outputs" / "tensor_ksh_dedup.npy"
DIMS_PATH = BASE_DIR / "ksh_outputs" / "tensor_ksh_dedup_dimensions.json"
CALIBRATION_PATH = BASE_DIR / "ksh_outputs" / "calibration_table.xlsx"
TEAOR_X_BIN_PATH = BASE_DIR / "teaor_x_bin.xlsx"
SETTLEMENT_WEIGHTS_PATH = BASE_DIR / "ksh_outputs" / "settlement_weights.xlsx"
OUTPUT_IMPUTE_SUMMARY_XLSX = BASE_DIR / "imputation_summary_by_teaor_bin.xlsx"
OUTPUT_CSV = BASE_DIR / "generated_companies_calibrated.csv"
OUTPUT_SUMMARY_XLSX = BASE_DIR / "generation_summary_by_teaor.xlsx"

SEED = 12345
TARGET_TOTAL_WORKERS = 4_650_000
ENABLE_GLOBAL_WORKER_SCALING = True
TEAOR_TARGET_OVERRIDES = {
    "97": 4500,
}

BIN_ORDER = ["1-4", "5-9", "10-19", "20-49", "50-249", "250-1000"]

BIN_RANGES = {
    "1-4": (1, 4),
    "5-9": (5, 9),
    "10-19": (10, 19),
    "20-49": (20, 49),
    "50-249": (50, 249),
    "250-1000": (250, 1000),
}

BIN_PROFILE_PARAMS = {
    "1-4": {"decay": 0.55, "blend": 0.75},
    "5-9": {"decay": 0.40, "blend": 0.70},
    "10-19": {"decay": 0.28, "blend": 0.65},
    "20-49": {"decay": 0.16, "blend": 0.60},
    "50-249": {"decay": 0.08, "blend": 0.55},
    "250-1000": {"decay": 0.03, "blend": 0.50},
}

SMALL_BINS = {"1-4", "5-9", "10-19"}

SMALL_BIN_MONO_PARAMS = {
    "1-4": {"ratio": 0.55},
    "5-9": {"ratio": 0.68},
    "10-19": {"ratio": 0.80},
}

SMALL_BIN_NEAR_MIN_THRESHOLD = {
    "1-4": 0.20,
    "5-9": 0.25,
    "10-19": 0.35,
}

def _mean_from_probs(values: list[int], probs: list[float]) -> float:
    return sum(v * p for v, p in zip(values, probs))


def _decay_probs_for_bin(
    values: list[int],
    target_avg: float,
    bin_name: str,
    max_iter: int = 80,
) -> list[float]:
    """
    Egyszerűbb, stabilabb profile builder:
    - van egy decay komponens
    - van egy uniform komponens
    - a kettőt keverjük
    - a target_avg-et csak lazán közelítjük, nem erőltetjük túl
    """
    lo = values[0]
    hi = values[-1]

    params = BIN_PROFILE_PARAMS[bin_name]
    decay_strength = params["decay"]
    blend = params["blend"]

    # 1) decay alapú profil
    decay_raw = [math.exp(-decay_strength * (v - lo)) for v in values]
    decay_sum = sum(decay_raw)
    decay_probs = [x / decay_sum for x in decay_raw]

    # 2) uniform profil
    uniform_probs = [1.0 / len(values)] * len(values)

    # 3) kevert profil
    probs = [
        blend * d + (1.0 - blend) * u
        for d, u in zip(decay_probs, uniform_probs)
    ]

    # 4) normalizálás
    s = sum(probs)
    probs = [p / s for p in probs]

    # 5) target_avg szerinti enyhe döntés
    current_avg = _mean_from_probs(values, probs)
    mid = (lo + hi) / 2

    # ha a kívánt átlag közelebb van a bin tetejéhez, laposítsunk
    if target_avg > current_avg:
        tilt = min(0.35, (target_avg - current_avg) / max(1e-9, (hi - lo)))
        tilted = []
        for v, p in zip(values, probs):
            rel = (v - lo) / max(1, (hi - lo))
            tilted.append(p * (1.0 + tilt * rel))
        s = sum(tilted)
        probs = [p / s for p in tilted]

    # ha a kívánt átlag túl alacsony, picit meredekítsünk
    elif target_avg < current_avg:
        tilt = min(0.35, (current_avg - target_avg) / max(1e-9, (hi - lo)))
        tilted = []
        for v, p in zip(values, probs):
            rel = (v - lo) / max(1, (hi - lo))
            tilted.append(p * (1.0 + tilt * (1.0 - rel)))
        s = sum(tilted)
        probs = [p / s for p in tilted]

    return probs

def _integer_counts_from_probs(
    company_count: int,
    probs: list[float],
    values: list[int],
) -> list[int]:
    """
    Egész darabszámok a profilból úgy, hogy:
    - ne csak 1-2 érték kapjon mindent
    - a nagyobb valószínűségű értékek domináljanak
    """
    exact = [company_count * p for p in probs]
    counts = [int(math.floor(x)) for x in exact]

    missing = company_count - sum(counts)

    # a maradékot a legnagyobb tört rész kapja
    remainders = [(i, exact[i] - counts[i]) for i in range(len(exact))]
    remainders.sort(key=lambda x: x[1], reverse=True)

    for i in range(missing):
        counts[remainders[i][0]] += 1

    return counts

def _build_small_bin_counts(
    values: list[int],
    company_count: int,
    bin_name: str,
    target_sum: int,
    rng: random.Random,
) -> list[int]:
    """
    Kis bin-ekhez budget-aware, monoton csökkenő count-vektort épít.

    Logika:
    - indulunk a minimum size teljes tömegével
    - megnézzük, mennyi extra worker-budget van
    - ebből eldől, hány size fér bele egyáltalán
    - csak az első K size lesz aktív
    - az aktív size-okon belül monoton csökkenő geometriai eloszlást építünk
    """
    width = len(values)
    lo = values[0]
    ratio = SMALL_BIN_MONO_PARAMS[bin_name]["ratio"]

    min_total = company_count * lo
    extra_budget = target_sum - min_total

    if extra_budget < 0:
        raise ValueError(
            f"Small bin target smaller than minimum: bin={bin_name}, "
            f"target={target_sum}, min_total={min_total}"
        )

    # 1) Meghatározzuk, hány size fér bele egyáltalán.
    # Ahhoz, hogy az első K size mind megjelenjen legalább egyszer,
    # az extra költség:
    # sum((values[i] - lo) for i in 1..K-1)
    active_width = 1
    needed = 0

    for k in range(2, width + 1):
        add_cost = values[k - 1] - lo
        if needed + add_cost <= extra_budget:
            needed += add_cost
            active_width = k
        else:
            break

    active_values = values[:active_width]

    # 2) Ha csak 1 size fér bele, mindenki minimum marad
    if active_width == 1:
        return [company_count] + [0] * (width - 1)

    # 3) Minden aktív size kap legalább 1-et
    base_counts = [1] * active_width
    remaining_companies = company_count - active_width

    # Az aktív size-ok minimális worker összege
    active_base_sum = sum(active_values)

    # A fennmaradó cégek alapból minimum size-on állnak
    current_sum = active_base_sum + remaining_companies * lo

    # Még felhasználható extra budget
    remaining_budget = target_sum - current_sum
    if remaining_budget < 0:
        # nagyon ritka eset, biztonság kedvéért
        remaining_budget = 0

    # 4) Geometriai csökkenő eloszlás az aktív size-okra
    weights = [ratio ** i for i in range(active_width)]
    wsum = sum(weights)
    probs = [w / wsum for w in weights]

    exact = [remaining_companies * p for p in probs]
    extra_counts = [int(math.floor(x)) for x in exact]

    missing = remaining_companies - sum(extra_counts)
    remainders = [(i, exact[i] - extra_counts[i]) for i in range(active_width)]
    remainders.sort(key=lambda x: x[1], reverse=True)

    for i in range(missing):
        extra_counts[remainders[i][0]] += 1

    counts_active = [b + e for b, e in zip(base_counts, extra_counts)]

    # 5) Monoton csökkenés kikényszerítése
    for i in range(1, active_width):
        if counts_active[i] > counts_active[i - 1]:
            counts_active[i] = counts_active[i - 1]

    # Ha emiatt elveszett pár cég, balról jobbra pótoljuk
    deficit = company_count - sum(counts_active)
    while deficit > 0:
        placed = False
        for i in range(active_width):
            if i == 0 or counts_active[i] + 1 <= counts_active[i - 1]:
                counts_active[i] += 1
                deficit -= 1
                placed = True
                break
        if not placed:
            counts_active[0] += 1
            deficit -= 1

    # 6) Enyhe upward correction, ha még túl alacsony az összeg.
    # Csak aktív size-ok között mozgunk, és megtartjuk a monotonitást.
    current_sum = sum(v * c for v, c in zip(active_values, counts_active))
    diff = target_sum - current_sum

    while diff > 0:
        moved = False

        # jobbról balra próbálunk emelni, mert így kisebb eltolással nő az összeg
        for i in range(active_width - 2, -1, -1):
            # egy céget áttolunk i -> i+1 irányba, ha marad monoton
            if counts_active[i] <= 0:
                continue

            new_left = counts_active[i] - 1
            new_right = counts_active[i + 1] + 1

            left_ok = True
            right_ok = True

            if i > 0:
                left_ok = counts_active[i - 1] >= new_left
            if i + 1 < active_width - 1:
                right_ok = new_right >= counts_active[i + 2]

            if left_ok and right_ok and new_left >= new_right:
                counts_active[i] = new_left
                counts_active[i + 1] = new_right
                diff -= 1
                moved = True

                if diff <= 0:
                    break

        if not moved:
            break

    # 7) Teljes width-re kiterjesztjük 0-kkal
    counts = counts_active + [0] * (width - active_width)

    return counts

def _adjust_counts_to_exact_sum(
    values: list[int],
    counts: list[int],
    target_sum: int,
    tolerance: int = 500,
) -> list[int]:
    """
    Lazább korrekció:
    - nem akarjuk mindenáron tűpontosan elérni a targetet
    - ha a különbség kicsi, békén hagyjuk az eloszlást
    """
    current_sum = sum(v * c for v, c in zip(values, counts))
    diff = target_sum - current_sum

    if abs(diff) <= tolerance:
        return counts

    value_to_idx = {v: i for i, v in enumerate(values)}

    # felfelé korrekció
    while diff > tolerance:
        moved = False
        movable = []
        for v in values[:-1]:
            i = value_to_idx[v]
            if counts[i] > 0:
                movable.append((counts[i], v))

        movable.sort(reverse=True)

        for _, v in movable:
            i = value_to_idx[v]
            j = value_to_idx[v + 1]
            if counts[i] > 0:
                counts[i] -= 1
                counts[j] += 1
                diff -= 1
                moved = True
                if diff <= tolerance:
                    break

        if not moved:
            break

    # lefelé korrekció
    while diff < -tolerance:
        moved = False
        movable = []
        for v in reversed(values[1:]):
            i = value_to_idx[v]
            if counts[i] > 0:
                movable.append((counts[i], v))

        movable.sort(reverse=True)

        for _, v in movable:
            i = value_to_idx[v]
            j = value_to_idx[v - 1]
            if counts[i] > 0:
                counts[i] -= 1
                counts[j] += 1
                diff += 1
                moved = True
                if diff >= -tolerance:
                    break

        if not moved:
            break

    return counts

def normalize_teaor_code(value) -> str | None:
    if pd.isna(value):
        return None

    s = str(value).strip()

    if s.endswith(".0"):
        s = s[:-2]

    s = s.lstrip("0")
    return s if s else "0"


def load_target_workers_by_teaor(
    calibration_path: Path,
    overrides: dict[str, int] | None = None,
) -> dict[str, int]:
    df = pd.read_excel(calibration_path)

    df["teaor_alag"] = df["teaor_alag"].map(normalize_teaor_code)
    df["ksh_worker_total"] = pd.to_numeric(df["ksh_worker_total"], errors="coerce")

    df = df[df["teaor_alag"].notna()].copy()
    df = df[df["ksh_worker_total"].notna()].copy()

    target_workers = {
        row["teaor_alag"]: int(round(row["ksh_worker_total"]))
        for _, row in df.iterrows()
    }

    if overrides:
        for teaor, value in overrides.items():
            target_workers[str(teaor)] = int(value)

    return target_workers


def aggregate_counts(counts: dict) -> tuple[dict[tuple[str, str], int], dict[str, int], dict[tuple[str, str], list[tuple[object, int]]]]:
    counts_by_teaor_bin = defaultdict(int)
    counts_by_teaor = defaultdict(int)
    cells_by_teaor_bin = defaultdict(list)

    for key, value in counts.items():
        teaor = str(key.teaor)
        bin_name = str(key.bin_name)

        counts_by_teaor_bin[(teaor, bin_name)] += int(value)
        counts_by_teaor[teaor] += int(value)
        cells_by_teaor_bin[(teaor, bin_name)].append((key, int(value)))

    return counts_by_teaor_bin, counts_by_teaor, cells_by_teaor_bin

def allocate_workers_to_cells_in_bin(
    cells: list[tuple[object, int]],
    total_target_workers: int,
    bin_name: str,
) -> dict[object, int]:
    """
    A bin teljes worker targetjét cellákra osztja szét.
    Minden cella kap:
    - minimumot: count_in_cell * lo
    - plusz workert arányosan a cellában lévő cégek számával
    """
    lo, hi = BIN_RANGES[bin_name]

    min_total = sum(count * lo for _, count in cells)
    max_total = sum(count * hi for _, count in cells)

    if total_target_workers < min_total or total_target_workers > max_total:
        raise ValueError(
            f"Cell-level target kívül esik a bin tartományán: "
            f"bin={bin_name}, target={total_target_workers}, "
            f"min_total={min_total}, max_total={max_total}"
        )

    # alap: minden cég minimum méreten indul
    result = {key: count * lo for key, count in cells}

    extra_target = total_target_workers - min_total
    if extra_target == 0:
        return result

    # Mivel minden cella ugyanabban a binben van,
    # a plusz kapacitás arányos a cégszámmal.
    exact_extra = []
    for key, count in cells:
        share = extra_target * (count / sum(c for _, c in cells))
        base = int(math.floor(share))
        result[key] += base
        exact_extra.append((key, share - base))

    missing = total_target_workers - sum(result.values())

    exact_extra.sort(key=lambda x: x[1], reverse=True)

    for i in range(missing):
        key = exact_extra[i][0]
        result[key] += 1

    return result

def _increase_sizes_greedily(
    sizes: list[int],
    hi: int,
    amount: int,
) -> int:
    """
    Növeli a size-okat legfeljebb `amount` workerrel.
    A kisebb size-okat emeli először, hogy a lecsengő shape kevésbé sérüljön.
    Visszatér: ténylegesen mennyit sikerült hozzáadni.
    """
    added = 0

    # mindig a legkisebbeket próbáljuk emelni először
    sizes.sort()

    while added < amount:
        moved = False
        for i in range(len(sizes)):
            if sizes[i] < hi:
                sizes[i] += 1
                added += 1
                moved = True
                if added >= amount:
                    break
        if not moved:
            break

    return added


def _decrease_sizes_greedily(
    sizes: list[int],
    lo: int,
    amount: int,
) -> int:
    """
    Csökkenti a size-okat legfeljebb `amount` workerrel.
    A legnagyobb size-okat csökkenti először.
    Visszatér: ténylegesen mennyit sikerült elvenni.
    """
    removed = 0

    # mindig a legnagyobbakat próbáljuk visszavenni először
    sizes.sort(reverse=True)

    while removed < amount:
        moved = False
        for i in range(len(sizes)):
            if sizes[i] > lo:
                sizes[i] -= 1
                removed += 1
                moved = True
                if removed >= amount:
                    break
        if not moved:
            break

    return removed

def allocate_bin_targets_for_teaor(
    teaor: str,
    counts_by_teaor_bin: dict[tuple[str, str], int],
    target_workers: int,
) -> tuple[dict[str, int], dict]:
    n_by_bin = {b: int(counts_by_teaor_bin.get((teaor, b), 0)) for b in BIN_ORDER}

    min_total = 0
    max_total = 0
    exact_targets = {}

    for b in BIN_ORDER:
        n = n_by_bin[b]
        lo, hi = BIN_RANGES[b]
        min_total += n * lo
        max_total += n * hi

    if max_total < min_total:
        raise ValueError(f"Invalid min/max for TEÁOR {teaor}")

    used_target = max(min_total, min(target_workers, max_total))

    total_capacity = max_total - min_total
    if total_capacity == 0:
        x = 0.0
    else:
        x = (used_target - min_total) / total_capacity

    for b in BIN_ORDER:
        n = n_by_bin[b]
        lo, hi = BIN_RANGES[b]
        exact_targets[b] = n * (lo + x * (hi - lo))

    floored = {}
    remainders = []
    sum_floor = 0

    for b in BIN_ORDER:
        n = n_by_bin[b]
        lo, hi = BIN_RANGES[b]
        min_bin = n * lo
        max_bin = n * hi

        base = int(math.floor(exact_targets[b]))
        base = max(min_bin, min(base, max_bin))

        floored[b] = base
        sum_floor += base
        remainders.append((b, exact_targets[b] - base))

    missing = used_target - sum_floor
    remainders.sort(key=lambda x: x[1], reverse=True)

    idx = 0
    while missing > 0 and idx < len(remainders):
        b = remainders[idx][0]
        n = n_by_bin[b]
        lo, hi = BIN_RANGES[b]
        max_bin = n * hi

        if floored[b] < max_bin:
            floored[b] += 1
            missing -= 1
        else:
            idx += 1

        if idx >= len(remainders) and missing > 0:
            idx = 0

    info = {
        "teaor": teaor,
        "target_workers_input": int(target_workers),
        "target_workers_used": int(used_target),
        "min_possible_workers": int(min_total),
        "max_possible_workers": int(max_total),
    }

    return floored, info


def build_company_sizes_for_bin(
    company_count: int,
    target_workers_in_bin: int,
    bin_name: str,
    rng: random.Random,
) -> list[int]:
    if company_count == 0:
        return []

    lo, hi = BIN_RANGES[bin_name]
    values = list(range(lo, hi + 1))

    min_total = company_count * lo
    max_total = company_count * hi

    if target_workers_in_bin < min_total or target_workers_in_bin > max_total:
        raise ValueError(
            f"Bin target kívül esik a tartományon. "
            f"bin={bin_name}, companies={company_count}, "
            f"target={target_workers_in_bin}, min={min_total}, max={max_total}"
        )

    # ---------- KIS BIN-EK: külön szabály ----------
    # Csak akkor használjuk, ha a cégszám legalább akkora, mint a bin szélessége.
    # Ha kevesebb a cég, mint a lehetséges size-ok száma, akkor ez a speciális logika
    # inkább szétszedi a shape-et, ezért visszaesünk az általános ágra.
    if bin_name in SMALL_BINS and company_count >= len(values):
        target_avg = target_workers_in_bin / company_count
        near_min_threshold = SMALL_BIN_NEAR_MIN_THRESHOLD[bin_name]

        # Ha a célátlag nagyon közel van a minimumhoz,
        # akkor ne próbáljunk shape-et építeni.
        if target_avg <= lo + near_min_threshold:
            sizes = [lo] * company_count
            rng.shuffle(sizes)
            return sizes

        counts = _build_small_bin_counts(
            values=values,
            company_count=company_count,
            bin_name=bin_name,
            target_sum=target_workers_in_bin,
            rng=rng,
        )

        # Kis bin-ekben most a shape az elsődleges,
        # ezért itt nem futtatunk külön exact-sum korrekciót.
        counts = counts

        sizes: list[int] = []
        for v, c in zip(values, counts):
            if c > 0:
                sizes.extend([v] * c)

        if len(sizes) != company_count:
            raise ValueError(
                f"Nem egyezik a cégek száma a kis binben: "
                f"bin={bin_name}, expected={company_count}, actual={len(sizes)}"
            )

        final_sum = sum(sizes)
        allowed_diff = max(5000, company_count)

        if abs(final_sum - target_workers_in_bin) > allowed_diff:
            raise ValueError(
                f"Túl nagy eltérés maradt a kis bin összegében: "
                f"bin={bin_name}, target={target_workers_in_bin}, actual={final_sum}, "
                f"allowed={allowed_diff}"
            )

        rng.shuffle(sizes)
        return sizes

    # ---------- NAGY BIN-EK: marad a profil-alapú módszer ----------
    target_avg = target_workers_in_bin / company_count
    mid = (lo + hi) / 2
    target_avg = 0.85 * target_avg + 0.15 * mid

    probs = _decay_probs_for_bin(values, target_avg, bin_name)

    counts = _integer_counts_from_probs(company_count, probs, values)

    counts = _adjust_counts_to_exact_sum(
        values,
        counts,
        target_workers_in_bin,
        tolerance=500,
    )

    sizes: list[int] = []
    for v, c in zip(values, counts):
        if c > 0:
            sizes.extend([v] * c)

    if len(sizes) != company_count:
        raise ValueError(
            f"Nem egyezik a cégek száma a binben: "
            f"bin={bin_name}, expected={company_count}, actual={len(sizes)}"
        )

    final_sum = sum(sizes)
    if abs(final_sum - target_workers_in_bin) > 500:
        raise ValueError(
            f"Túl nagy eltérés maradt az összegben: {bin_name}, "
            f"target={target_workers_in_bin}, actual={final_sum}"
        )

    rng.shuffle(sizes)
    return sizes

def generate_companies_calibrated(
    counts: dict,
    target_workers_by_teaor: dict[str, int],
    seed: int = 12345,
) -> tuple[dict, pd.DataFrame]:
    rng = random.Random(seed)

    counts_by_teaor_bin, counts_by_teaor, cells_by_teaor_bin = aggregate_counts(counts)

    generated = {}
    summary_rows = []

    all_teaors = sorted({teaor for teaor, _ in counts_by_teaor_bin.keys()}, key=lambda x: int(x))

    for teaor in all_teaors:
        target_workers = target_workers_by_teaor.get(teaor)

        if target_workers is None:
            # fallback: bin-közepekkel becslünk
            fallback = 0
            for b in BIN_ORDER:
                n = counts_by_teaor_bin.get((teaor, b), 0)
                lo, hi = BIN_RANGES[b]
                fallback += int(round(n * ((lo + hi) / 2)))
            target_workers = fallback

        bin_targets, info = allocate_bin_targets_for_teaor(
            teaor=teaor,
            counts_by_teaor_bin=counts_by_teaor_bin,
            target_workers=target_workers,
        )

        actual_workers_for_teaor = 0
        actual_companies_for_teaor = 0

        for b in BIN_ORDER:
            total_companies_in_bin = counts_by_teaor_bin.get((teaor, b), 0)
            total_workers_in_bin = bin_targets[b]

            cells = cells_by_teaor_bin.get((teaor, b), [])

            cell_targets = allocate_workers_to_cells_in_bin(
                cells=cells,
                total_target_workers=total_workers_in_bin,
                bin_name=b,
            )

            lo, hi = BIN_RANGES[b]

            # 1) első kör: cellánként generálunk
            bin_parts = {}
            bin_worker_sum = 0
            bin_company_sum = 0

            for key, count_in_cell in cells:
                cell_target_workers = cell_targets[key]

                part = build_company_sizes_for_bin(
                    company_count=count_in_cell,
                    target_workers_in_bin=cell_target_workers,
                    bin_name=b,
                    rng=rng,
                )

                bin_parts[key] = part
                bin_worker_sum += sum(part)
                bin_company_sum += len(part)

            # 2) bin-szintű reconciliation: hozzuk vissza a teljes worker összeget
            diff = total_workers_in_bin - bin_worker_sum

            if diff > 0:
                # kevés worker lett -> emeljük a size-okat ott, ahol van headroom
                # kisebb átlagú cellák előnyt kapnak
                ordered_keys = sorted(
                    bin_parts.keys(),
                    key=lambda k: (sum(bin_parts[k]) / max(1, len(bin_parts[k])))
                )

                for key in ordered_keys:
                    if diff <= 0:
                        break
                    added = _increase_sizes_greedily(bin_parts[key], hi=hi, amount=diff)
                    diff -= added

            elif diff < 0:
                # túl sok worker lett -> vegyünk vissza
                diff = -diff
                ordered_keys = sorted(
                    bin_parts.keys(),
                    key=lambda k: (sum(bin_parts[k]) / max(1, len(bin_parts[k]))),
                    reverse=True
                )

                for key in ordered_keys:
                    if diff <= 0:
                        break
                    removed = _decrease_sizes_greedily(bin_parts[key], lo=lo, amount=diff)
                    diff -= removed

            # 3) végső mentés
            final_bin_worker_sum = 0
            final_bin_company_sum = 0

            for key, part in bin_parts.items():
                generated[key] = part
                final_bin_worker_sum += sum(part)
                final_bin_company_sum += len(part)

            actual_workers_for_teaor += final_bin_worker_sum
            actual_companies_for_teaor += final_bin_company_sum

        summary_rows.append(
            {
                "teaor": teaor,
                "company_count_total": actual_companies_for_teaor,
                "target_workers_input": info["target_workers_input"],
                "target_workers_used": info["target_workers_used"],
                "actual_generated_workers": actual_workers_for_teaor,
                "min_possible_workers": info["min_possible_workers"],
                "max_possible_workers": info["max_possible_workers"],
                "target_avg_size_used": (
                    info["target_workers_used"] / actual_companies_for_teaor
                    if actual_companies_for_teaor > 0 else None
                ),
                "actual_avg_size": (
                    actual_workers_for_teaor / actual_companies_for_teaor
                    if actual_companies_for_teaor > 0 else None
                ),
                "difference_actual_minus_used_target": actual_workers_for_teaor - info["target_workers_used"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("teaor")

    print("\n--- Allokációs összegzés ---")
    print(f"Summary target_workers_input total: {int(summary_df['target_workers_input'].sum()):,}")
    print(f"Summary target_workers_used total: {int(summary_df['target_workers_used'].sum()):,}")
    print(f"Summary actual_generated_workers total: {int(summary_df['actual_generated_workers'].sum()):,}")

    clipped_df = summary_df[
        summary_df["target_workers_input"] != summary_df["target_workers_used"]
        ].copy()

    print(f"Levágott (clipped) TEÁOR-ok száma: {len(clipped_df):,}")

    if not clipped_df.empty:
        print("\nElső 20 levágott TEÁOR:")
        print(
            clipped_df[
                [
                    "teaor",
                    "target_workers_input",
                    "target_workers_used",
                    "min_possible_workers",
                    "max_possible_workers",
                    "company_count_total",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    return generated, summary_df


def flatten_generated(generated: dict) -> list[dict]:
    rows = []

    for key, sizes in generated.items():
        for idx, size in enumerate(sizes):
            rows.append(
                {
                    "settlement": key.settlement,
                    "teaor": key.teaor,
                    "bin": key.bin_name,
                    "company_index_in_cell": idx,
                    "company_size": int(size),
                }
            )

    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    counts = load_counts_from_tensor_files(
        tensor_npy_path=str(TENSOR_PATH),
        dimensions_json_path=str(DIMS_PATH),
        drop_zeros=True,
    )

    original_company_total = sum(counts.values())

    counts, impute_summary_df = impute_counts_to_national_targets(
        counts=counts,
        teaor_x_bin_path=TEAOR_X_BIN_PATH,
        settlement_weights_path=SETTLEMENT_WEIGHTS_PATH,
    )

    imputed_company_total = sum(counts.values())

    print(f"Original company total from tensor: {original_company_total:,}")
    print(f"Company total after imputation:     {imputed_company_total:,}")
    print(
        f"Imputed extra companies:            "
        f"{imputed_company_total - original_company_total:,}"
    )

    impute_summary_df.to_excel(
        OUTPUT_IMPUTE_SUMMARY_XLSX,
        index=False,
        engine="openpyxl",
    )
    print(f"Wrote: {OUTPUT_IMPUTE_SUMMARY_XLSX}")

    target_workers_by_teaor = load_target_workers_by_teaor(
        CALIBRATION_PATH,
        overrides=TEAOR_TARGET_OVERRIDES,
    )

    original_target_total = sum(target_workers_by_teaor.values())

    if ENABLE_GLOBAL_WORKER_SCALING and original_target_total > 0:
        scale_factor = TARGET_TOTAL_WORKERS / original_target_total
        target_workers_by_teaor = {
            teaor: int(round(value * scale_factor))
            for teaor, value in target_workers_by_teaor.items()
        }
        scaled_target_total = sum(target_workers_by_teaor.values())

        print(f"Global worker scaling enabled: True")
        print(f"Original target worker total: {original_target_total:,}")
        print(f"Scaled target worker total:   {scaled_target_total:,}")
        print(f"Scale factor: {scale_factor:.6f}")
    else:
        print("Global worker scaling enabled: False")
        print(f"Target worker total: {original_target_total:,}")

    print(f"Loaded non-zero cells: {len(counts):,}")
    print(f"TEÁOR target workers loaded: {len(target_workers_by_teaor):,}")
    print(f"Loaded target workers total: {sum(target_workers_by_teaor.values()):,}")

    teaors_in_counts = sorted({str(k.teaor) for k in counts.keys()}, key=lambda x: int(x))
    teaors_in_targets = sorted(target_workers_by_teaor.keys(), key=lambda x: int(x))

    missing_in_targets = sorted(set(teaors_in_counts) - set(teaors_in_targets), key=lambda x: int(x))
    missing_in_counts = sorted(set(teaors_in_targets) - set(teaors_in_counts), key=lambda x: int(x))

    print(f"TEÁOR-ok a counts-ban: {len(teaors_in_counts):,}")
    print(f"TEÁOR-ok a target táblában: {len(teaors_in_targets):,}")
    print(f"Hiányzik a targetből: {missing_in_targets}")
    print(f"Hiányzik a counts-ból: {missing_in_counts}")

    generated, summary_df = generate_companies_calibrated(
        counts=counts,
        target_workers_by_teaor=target_workers_by_teaor,
        seed=SEED,
    )

    total_companies = sum(len(v) for v in generated.values())
    total_workers = sum(sum(v) for v in generated.values())

    print(f"Generated companies: {total_companies:,}")
    print(f"Generated workers: {total_workers:,}")

    rows = flatten_generated(generated)
    write_csv(rows, OUTPUT_CSV)
    summary_df.to_excel(OUTPUT_SUMMARY_XLSX, index=False, engine="openpyxl")

    print(f"Mentve: {OUTPUT_CSV}")
    print(f"Mentve: {OUTPUT_SUMMARY_XLSX}")


if __name__ == "__main__":
    main()