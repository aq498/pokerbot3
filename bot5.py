import os
import sys
import signal

# Lock glibc & Python arena allocators BEFORE spawning worker processes to prevent postflop RAM spikes
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")
os.environ.setdefault("MALLOC_MMAP_THRESHOLD_", "65536")
os.environ.setdefault("PYTHONMALLOC", "malloc")

import asyncio
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import inspect
import json
import math
import random
import ctypes
import gc
import multiprocessing as mp
import sqlite3
import time
import uuid
from collections import defaultdict, Counter

import aiohttp
from aiohttp import web
import requests
import websockets
from treys import Card, Evaluator, Deck

# Force aggressive garbage collection thresholds to prevent arena bloat
gc.set_threshold(400, 5, 5)


def trim_process_memory():
    """Forces Python GC and Linux glibc malloc_trim(0) to release freed heap pages back to OS."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


# =====================================================================
# 1. CONFIGURATION & ENVIRONMENT (4-Core Optimized)
# =====================================================================
API_KEY = os.getenv("OPEN_POKER_API_KEY", "")
WS_URL = os.getenv("OPEN_POKER_WS_URL", "wss://openpoker.ai/ws")
REST_BASE_URL = os.getenv("OPEN_POKER_REST_URL", "https://openpoker.ai")
PORT = int(os.getenv("PORT", 10000))
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "")
DB_PATH = os.getenv("POKER_DB_PATH", "opponent_memory.db")

# 4-Core Adaptive Timing Config
CPU_WORKERS = int(os.getenv("CPU_WORKERS", "4"))
TARGET_ACTION_TIME = float(os.getenv("TARGET_ACTION_TIME", "4.75"))
MIN_ADAPTIVE_TIME = float(os.getenv("MIN_ADAPTIVE_TIME", "0.60"))
MC_MAX_ITERS = int(os.getenv("MC_MAX_ITERS", "200000"))
MAX_RAM_PROFILES = int(os.getenv("MAX_RAM_PROFILES", "500"))

# Stack Management & Ratholing (Leave at 5,000, Rejoin with 1,500)
BUY_IN_AMOUNT = int(os.getenv("BUY_IN_AMOUNT", "1500"))
PROFIT_LOCK_STACK = int(os.getenv("PROFIT_LOCK_STACK", "5000"))
TOURNAMENT_MODE = os.getenv("TOURNAMENT_MODE", "false").lower() == "true"

FULL_DECK_INTS = Deck.GetFullDeck()
RANK_CHARS = "23456789TJQKA"

# =====================================================================
# 2. MECHANISM #1: PRE-COMPUTED GTO RANGES & TABULAR CFR SOLVER
# =====================================================================
PREFLOP_169_LOOKUP = {
    "AA": 100, "KK": 98, "QQ": 95, "JJ": 91, "TT": 87, "99": 82, "88": 78,
    "77": 74, "66": 70, "55": 67, "44": 64, "33": 62, "22": 60,
    "AKs": 96, "AQs": 92, "AJs": 88, "ATs": 85, "A9s": 76, "A8s": 74, "A7s": 72,
    "A6s": 70, "A5s": 75, "A4s": 73, "A3s": 71, "A2s": 69,
    "KQs": 89, "KJs": 85, "KTs": 81, "K9s": 73, "K8s": 66, "K7s": 64, "K6s": 62, "K5s": 60,
    "QJs": 83, "QTs": 79, "Q9s": 71, "Q8s": 63,
    "JTs": 82, "J9s": 72, "J8s": 64,
    "T9s": 76, "T8s": 68, "98s": 73, "97s": 64, "87s": 71, "86s": 62,
    "76s": 69, "75s": 60, "65s": 67, "54s": 65,
    "AKo": 93, "AQo": 86, "AJo": 81, "ATo": 75, "A9o": 64, "A8o": 61, "A7o": 58, "A5o": 57,
    "KQo": 82, "KJo": 76, "KTo": 71, "K9o": 60,
    "QJo": 74, "QTo": 68, "JTo": 70, "T9o": 61, "98o": 58,
}

GTO_OPEN_MIX = {
    "UTG": {"min_pure": 80, "min_mix": 73},
    "MP":  {"min_pure": 76, "min_mix": 68},
    "CO":  {"min_pure": 69, "min_mix": 60},
    "BTN": {"min_pure": 62, "min_mix": 50},
    "SB":  {"min_pure": 64, "min_mix": 52},
    "BB":  {"min_pure": 82, "min_mix": 74},
}

THREE_BET_BLUFFS = {"A5s", "A4s", "A3s", "K9s", "Q9s", "J9s", "T9s", "98s", "87s"}
FOUR_BET_VALUE = {"AA", "KK", "QQ", "AKs", "AKo"}


def cards_to_hand_str(c1: int, c2: int) -> str:
    r1, r2 = Card.get_rank_int(c1), Card.get_rank_int(c2)
    s1, s2 = Card.get_suit_int(c1), Card.get_suit_int(c2)
    if r1 == r2:
        return f"{RANK_CHARS[r1]}{RANK_CHARS[r2]}"
    high, low = (r1, r2) if r1 > r2 else (r2, r1)
    suffix = "s" if s1 == s2 else "o"
    return f"{RANK_CHARS[high]}{RANK_CHARS[low]}{suffix}"


def get_preflop_score(hand_str: str) -> int:
    if hand_str in PREFLOP_169_LOOKUP:
        return PREFLOP_169_LOOKUP[hand_str]
    r1 = RANK_CHARS.index(hand_str[0])
    r2 = RANK_CHARS.index(hand_str[1])
    suited = hand_str.endswith("s")
    gap = r1 - r2
    base = r1 * 3.2 + r2 * 1.4 + (8 if suited else 0) - min(12, max(0, gap - 1) * 3)
    return int(max(10, min(65, base)))


def hand_preflop_tier(hand_str: str) -> int:
    score = get_preflop_score(hand_str)
    if score >= 87:
        return 1
    if score >= 78:
        return 2
    if score >= 67:
        return 3
    if score >= 58:
        return 4
    return 5


def get_gto_preflop_frequencies(position: str, hand_str: str, facing_open: bool, in_position: bool):
    score = get_preflop_score(hand_str)
    cfg = GTO_OPEN_MIX.get(position, GTO_OPEN_MIX["MP"])
    if not facing_open:
        if score >= cfg["min_pure"]:
            return (1.0, 0.0, 0.0)
        elif score >= cfg["min_mix"]:
            span = max(1, cfg["min_pure"] - cfg["min_mix"])
            p_r = (score - cfg["min_mix"]) / span
            return (round(p_r, 3), round(1.0 - p_r, 3) if position in ("SB", "BB") else 0.0, round(1.0 - p_r, 3))
        return (0.0, 0.0, 1.0)
    else:
        if score >= 90:
            return (0.85, 0.15, 0.0)
        if score >= 83:
            return (0.45 if in_position else 0.55, 0.55 if in_position else 0.45, 0.0)
        if hand_str in THREE_BET_BLUFFS:
            return (0.28 if in_position else 0.18, 0.42, 0.30)
        if score >= 68:
            return (0.05, 0.75 if in_position else 0.50, 0.20 if in_position else 0.45)
        return (0.0, 0.0, 1.0)


class LightweightSubgameCFR:
    @staticmethod
    def solve_postflop_mix(adj_equity: float, pot: float, to_call: float, candidate_evs: list, blocker_score: float):
        n_actions = 2 + len(candidate_evs)
        regret_sum = [0.0] * n_actions
        strategy_sum = [0.0] * n_actions

        ev_fold_or_check = 0.0 if to_call > 0 else adj_equity * pot
        ev_call = (adj_equity * (pot + to_call) - to_call) if to_call > 0 else ev_fold_or_check

        base_evs = [ev_fold_or_check, ev_call] + [ev for _, ev in candidate_evs]
        for idx in range(2, n_actions):
            base_evs[idx] += blocker_score * (pot * 0.08)

        scale = max(1.0, pot + to_call)
        norm_evs = [v / scale for v in base_evs]

        for t in range(1, 61):
            pos_regrets = [max(0.0, r) for r in regret_sum]
            r_total = sum(pos_regrets)
            if r_total > 0:
                strat = [r / r_total for r in pos_regrets]
            else:
                strat = [1.0 / n_actions] * n_actions

            node_util = sum(strat[a] * norm_evs[a] for a in range(n_actions))
            for a in range(n_actions):
                inst_regret = norm_evs[a] - node_util
                regret_sum[a] = max(0.0, regret_sum[a] + inst_regret)
                strategy_sum[a] += t * strat[a]

        total_strat = sum(strategy_sum)
        if total_strat <= 0:
            return [1.0 / n_actions] * n_actions
        return [s / total_strat for s in strategy_sum]


def _build_range_tiers():
    all_169 = set()
    for i in range(13):
        all_169.add(f"{RANK_CHARS[i]}{RANK_CHARS[i]}")
        for j in range(i):
            all_169.add(f"{RANK_CHARS[i]}{RANK_CHARS[j]}s")
            all_169.add(f"{RANK_CHARS[i]}{RANK_CHARS[j]}o")
    return {
        t: frozenset(h for h in all_169 if get_preflop_score(h) >= min_sc)
        for t, min_sc in [(1, 87), (2, 78), (3, 67), (4, 58)]
    }


RANGE_TIER = _build_range_tiers()
RANGE_VS_3BET = frozenset(h for h in RANGE_TIER[3] if get_preflop_score(h) >= 81)


# =====================================================================
# 3. MECHANISM #4: BLOCKERS & CARD REMOVAL NUANCES
# =====================================================================
def compute_blocker_effects(hole_cards, community_cards):
    res = {
        "blocks_nut_flush": False,
        "blocks_second_nut_flush": False,
        "blocks_board_straight": False,
        "blocks_top_pair": False,
        "unblocks_folding_range": False,
        "blocker_score": 0.0,
    }
    if len(hole_cards) < 2 or len(community_cards) < 3:
        return res

    board_ranks = [Card.get_rank_int(c) for c in community_cards]
    board_suits = [Card.get_suit_int(c) for c in community_cards]
    hole_ranks = [Card.get_rank_int(c) for c in hole_cards]
    hole_suits = [Card.get_suit_int(c) for c in hole_cards]

    score = 0.0
    suit_counts = Counter(board_suits)
    dom_suit, dom_count = suit_counts.most_common(1)[0]

    if dom_count >= 3:
        for c in hole_cards:
            if Card.get_suit_int(c) == dom_suit:
                rk = Card.get_rank_int(c)
                if rk == 12:
                    res["blocks_nut_flush"] = True
                    score += 0.22
                elif rk == 11:
                    res["blocks_second_nut_flush"] = True
                    score += 0.12

    max_b_rank = max(board_ranks)
    if max_b_rank in hole_ranks:
        res["blocks_top_pair"] = True
        score += 0.10

    median_b = sorted(board_ranks)[len(board_ranks) // 2]
    if any(abs(hr - median_b) <= 1 for hr in hole_ranks):
        res["blocks_board_straight"] = True
        score += 0.08

    if len(community_cards) == 5 and dom_count == 2:
        if dom_suit not in hole_suits:
            res["unblocks_folding_range"] = True
            score += 0.06

    res["blocker_score"] = max(-0.25, min(0.45, score))
    return res


# =====================================================================
# 4. BOARD TEXTURE, DRAW DETECTOR & COUNTERFEIT GUARD
# =====================================================================
def analyze_board_and_draws(hole_cards, community_cards, evaluator: Evaluator):
    info = {
        "paired": False, "double_paired": False, "flushy": False, "four_flush": False,
        "connected": False, "four_straight": False, "dry": True, "wet": False, "high": False,
        "hero_flush_draw": False, "hero_nut_flush_draw": False, "hero_oesd": False,
        "hero_gutshot": False, "draw_outs": 0, "playing_board": False,
        "vulnerable_one_pair": False, "rank_class": 9,
    }
    if len(community_cards) < 3 or len(hole_cards) < 2:
        return info

    board_ranks = [Card.get_rank_int(c) for c in community_cards]
    board_suits = [Card.get_suit_int(c) for c in community_cards]
    hole_ranks = [Card.get_rank_int(c) for c in hole_cards]
    hole_suits = [Card.get_suit_int(c) for c in hole_cards]

    rank_counts = Counter(board_ranks)
    info["paired"] = any(v >= 2 for v in rank_counts.values())
    info["double_paired"] = sum(1 for v in rank_counts.values() if v >= 2) >= 2 or any(v >= 3 for v in rank_counts.values())

    suit_counts = Counter(board_suits)
    _, max_board_suit = suit_counts.most_common(1)[0]
    info["flushy"] = max_board_suit >= 3
    info["four_flush"] = max_board_suit >= 4
    two_tone = max_board_suit == 2

    unique_b_ranks = sorted(set(board_ranks))
    if 12 in unique_b_ranks:
        unique_b_ranks = [-1] + unique_b_ranks
    max_in_span5 = 1
    for i in range(len(unique_b_ranks)):
        cnt = sum(1 for r in unique_b_ranks if 0 <= r - unique_b_ranks[i] <= 4)
        if cnt > max_in_span5:
            max_in_span5 = cnt

    info["connected"] = max_in_span5 >= 3
    info["four_straight"] = max_in_span5 >= 4
    info["high"] = max(board_ranks) >= 10
    info["wet"] = info["flushy"] or info["connected"] or (two_tone and max_in_span5 >= 2 and not info["paired"])
    info["dry"] = not info["wet"]

    score = evaluator.evaluate(community_cards, hole_cards)
    rc = evaluator.get_rank_class(score)
    info["rank_class"] = rc

    if len(community_cards) == 5:
        board_only_score = evaluator.evaluate(community_cards, [])
        if score == board_only_score:
            info["playing_board"] = True

    hero_matches_board = sum(1 for r in hole_ranks if r in board_ranks)
    is_pocket_pair = hole_ranks[0] == hole_ranks[1]
    if rc == 8:
        info["vulnerable_one_pair"] = True
    elif rc == 7 and info["paired"] and not is_pocket_pair and hero_matches_board == 1:
        info["vulnerable_one_pair"] = True
    elif rc == 7 and info["double_paired"] and not is_pocket_pair and hero_matches_board == 0:
        info["vulnerable_one_pair"] = True

    if len(community_cards) < 5:
        all_suits = Counter(board_suits + hole_suits)
        suit_id, suit_total = all_suits.most_common(1)[0]
        if suit_total == 4 and (suit_id in hole_suits):
            info["hero_flush_draw"] = True
            for c in hole_cards:
                if Card.get_suit_int(c) == suit_id and Card.get_rank_int(c) >= 11:
                    info["hero_nut_flush_draw"] = True

        all_ranks = set(board_ranks + hole_ranks)
        straight_completing_ranks = 0
        for test_r in range(13):
            if test_r in all_ranks:
                continue
            cand = all_ranks | {test_r}
            if 12 in cand:
                cand = cand | {-1}
            hole_set = set(hole_ranks) | ({-1} if 12 in hole_ranks else set())
            for low in range(-1, 9):
                window = {low, low + 1, low + 2, low + 3, low + 4}
                if window.issubset(cand) and (window & hole_set):
                    straight_completing_ranks += 1
                    break

        if straight_completing_ranks >= 2:
            info["hero_oesd"] = True
        elif straight_completing_ranks == 1:
            info["hero_gutshot"] = True

        outs = 0
        if info["hero_flush_draw"]:
            outs += 9
        if info["hero_oesd"]:
            outs += 6 if info["hero_flush_draw"] else 8
        elif info["hero_gutshot"]:
            outs += 3 if info["hero_flush_draw"] else 4
        info["draw_outs"] = outs

    return info


# =====================================================================
# 5. MECHANISM #3: BAYESIAN HAND WEIGHTING + 4-CPU ADAPTIVE MC WORKER
# =====================================================================
_WORKER_EVALUATOR = None


def _init_worker():
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = Evaluator()


def _adaptive_mc_worker(args):
    (
        hole_cards, community_cards, num_opponents, bayesian_weights,
        postflop_aggression, opp_bluff_freq, decision_thresholds,
        min_time, max_time, max_iters, seed,
    ) = args

    rng = random.Random(seed)
    evaluator = _WORKER_EVALUATOR or Evaluator()
    t_start = time.perf_counter()

    dead_cards = set(hole_cards) | set(community_cards)
    available = [c for c in FULL_DECK_INTS if c not in dead_cards]
    cards_to_come = 5 - len(community_cards)
    street = len(community_cards)

    valid_opp_combos = []
    n_avail = len(available)
    opp_pair_buf = [0, 0]
    for i in range(n_avail):
        c1 = available[i]
        bit_i = 1 << i
        for j in range(i + 1, n_avail):
            c2 = available[j]
            h_str = cards_to_hand_str(c1, c2)
            prior_w = bayesian_weights.get(h_str, 0.0) if bayesian_weights else 1.0
            if prior_w <= 0.01:
                continue

            w = prior_w
            if street >= 3 and postflop_aggression >= 1:
                opp_pair_buf[0] = c1
                opp_pair_buf[1] = c2
                opp_sc = evaluator.evaluate(community_cards, opp_pair_buf)
                opp_rc = evaluator.get_rank_class(opp_sc)
                if postflop_aggression >= 2:
                    likelihood = 1.0 if opp_rc <= 7 else (0.50 if opp_rc == 8 else max(0.10, opp_bluff_freq))
                else:
                    likelihood = 1.0 if opp_rc <= 8 else max(0.22, opp_bluff_freq * 1.2)
                w *= likelihood
            valid_opp_combos.append((bit_i | (1 << j), c1, c2, w))

    if not valid_opp_combos:
        for i in range(n_avail):
            for j in range(i + 1, n_avail):
                valid_opp_combos.append(((1 << i) | (1 << j), available[i], available[j], 1.0))

    if num_opponents == 1 and street == 5:
        my_score = evaluator.evaluate(community_cards, hole_cards)
        w_wins = 0.0
        w_total = 0.0
        for _mask, c1, c2, w in valid_opp_combos:
            opp_pair_buf[0] = c1
            opp_pair_buf[1] = c2
            opp_score = evaluator.evaluate(community_cards, opp_pair_buf)
            w_total += w
            if my_score < opp_score:
                w_wins += w
            elif my_score == opp_score:
                w_wins += 0.5 * w
        trim_process_memory()
        return w_wins, w_total, len(valid_opp_combos), True

    if num_opponents == 1 and street == 4 and len(valid_opp_combos) * n_avail <= 22000:
        w_wins = 0.0
        w_total = 0.0
        evals = 0
        sim_board = list(community_cards) + [0]
        for r_idx, river_card in enumerate(available):
            r_bit = 1 << r_idx
            sim_board[4] = river_card
            my_score = evaluator.evaluate(sim_board, hole_cards)
            for combo_mask, c1, c2, w in valid_opp_combos:
                if combo_mask & r_bit:
                    continue
                opp_pair_buf[0] = c1
                opp_pair_buf[1] = c2
                opp_score = evaluator.evaluate(sim_board, opp_pair_buf)
                w_total += w
                if my_score < opp_score:
                    w_wins += w
                elif my_score == opp_score:
                    w_wins += 0.5 * w
                evals += 1
        trim_process_memory()
        return w_wins, w_total, evals, True

    w_wins = 0.0
    w_total = 0.0
    iterations = 0
    n_combos = len(valid_opp_combos)
    early_stopped = False

    sim_board = list(community_cards) + [0] * cards_to_come
    opp_hand_buf = [[0, 0] for _ in range(num_opponents)]
    base_board_len = len(community_cards)

    while iterations < max_iters:
        now = time.perf_counter()
        elapsed = now - t_start
        if elapsed >= max_time:
            break

        if iterations >= 1500 and elapsed >= min_time and w_total > 0:
            p_hat = w_wins / w_total
            se = math.sqrt(max(0.0001, p_hat * (1.0 - p_hat)) / iterations)
            ci_radius = 2.576 * se + (0.008 / math.sqrt(CPU_WORKERS))
            is_ambiguous = any(abs(p_hat - th) <= ci_radius for th in decision_thresholds)
            if not is_ambiguous and ci_radius < 0.018:
                early_stopped = True
                break

        batch = min(256, max_iters - iterations)
        for _ in range(batch):
            sample_w = 1.0
            skip = False
            used_mask = 0
            for opp_idx in range(num_opponents):
                found = False
                for _try in range(8):
                    combo_mask, c1, c2, w = valid_opp_combos[rng.randrange(n_combos)]
                    if not (used_mask & combo_mask):
                        opp_hand_buf[opp_idx][0] = c1
                        opp_hand_buf[opp_idx][1] = c2
                        used_mask |= combo_mask
                        if opp_idx == 0:
                            sample_w = w
                        found = True
                        break
                if not found:
                    skip = True
                    break
            if skip:
                continue

            if cards_to_come > 0:
                filled = 0
                for _try in range(cards_to_come * 4):
                    r_idx = rng.randrange(n_avail)
                    b_rc = 1 << r_idx
                    if not (used_mask & b_rc):
                        used_mask |= b_rc
                        sim_board[base_board_len + filled] = available[r_idx]
                        filled += 1
                        if filled == cards_to_come:
                            break
                if filled < cards_to_come:
                    continue

            my_score = evaluator.evaluate(sim_board, hole_cards)
            best_opp_score = 7463
            for opp_idx in range(num_opponents):
                sc = evaluator.evaluate(sim_board, opp_hand_buf[opp_idx])
                if sc < best_opp_score:
                    best_opp_score = sc

            w_total += sample_w
            if my_score < best_opp_score:
                w_wins += sample_w
            elif my_score == best_opp_score:
                w_wins += 0.5 * sample_w
            iterations += 1

    trim_process_memory()
    return w_wins, w_total, iterations, early_stopped


# =====================================================================
# 6. MECHANISMS #6 & #7: PERSISTENT SQLITE DB + SHOWDOWN X-RAY PROFILER
# =====================================================================
class OpponentTracker:
    def __init__(self, db_path: str = DB_PATH, max_ram_profiles: int = MAX_RAM_PROFILES):
        self.db_path = db_path
        self.max_ram_profiles = max_ram_profiles
        self.evaluator = Evaluator()
        self.stats = OrderedDict()
        self.dirty_pids = set()
        self.hand_vpip_flagged = set()
        self.hand_pfr_flagged = set()
        self.hand_aggressors = set()
        self._init_sqlite()
        self._load_from_sqlite()

    @staticmethod
    def _default_profile():
        return {
            "hands": 0, "vpip": 0, "pfr": 0, "postflop_aggressive": 0, "postflop_passive": 0,
            "folds_to_bet": 0, "faced_bet": 0, "showdowns": 0, "showdown_bluffs": 0, "showdown_traps": 0,
        }

    def _get_or_create_profile(self, pid: str) -> dict:
        if pid in self.stats:
            self.stats.move_to_end(pid)
            return self.stats[pid]
        if len(self.stats) >= self.max_ram_profiles:
            evicted_pid, _ = self.stats.popitem(last=False)
            self.dirty_pids.discard(evicted_pid)
        prof = self._default_profile()
        self.stats[pid] = prof
        return prof

    def _init_sqlite(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS player_profiles (
                        player_id TEXT PRIMARY KEY,
                        hands INTEGER DEFAULT 0,
                        vpip INTEGER DEFAULT 0,
                        pfr INTEGER DEFAULT 0,
                        postflop_aggressive INTEGER DEFAULT 0,
                        postflop_passive INTEGER DEFAULT 0,
                        folds_to_bet INTEGER DEFAULT 0,
                        faced_bet INTEGER DEFAULT 0,
                        showdowns INTEGER DEFAULT 0,
                        showdown_bluffs INTEGER DEFAULT 0,
                        showdown_traps INTEGER DEFAULT 0,
                        updated_at REAL
                    )
                    """
                )
                conn.commit()
        except Exception as e:
            print(f"[SQLite] Init warning: {e}")

    def _load_from_sqlite(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT * FROM player_profiles ORDER BY updated_at DESC LIMIT ?",
                    (self.max_ram_profiles,),
                )
                rows = cur.fetchall()
                for r in reversed(rows):
                    pid = r[0]
                    self.stats[pid] = {
                        "hands": r[1], "vpip": r[2], "pfr": r[3], "postflop_aggressive": r[4],
                        "postflop_passive": r[5], "folds_to_bet": r[6], "faced_bet": r[7],
                        "showdowns": r[8], "showdown_bluffs": r[9], "showdown_traps": r[10],
                    }
            print(f"[SQLite] Loaded {len(self.stats)} persistent opponent profiles (LRU cap={self.max_ram_profiles}).")
        except Exception as e:
            print(f"[SQLite] Load warning: {e}")

    def flush_to_sqlite(self, pids=None):
        try:
            target_pids = list(pids) if pids else list(self.dirty_pids)
            if not target_pids:
                return
            now = time.time()
            rows_to_write = []
            for pid in target_pids:
                s = self.stats.get(pid)
                if s:
                    rows_to_write.append((
                        pid, s["hands"], s["vpip"], s["pfr"],
                        s["postflop_aggressive"], s["postflop_passive"],
                        s["folds_to_bet"], s["faced_bet"],
                        s["showdowns"], s["showdown_bluffs"], s["showdown_traps"], now,
                    ))
                self.dirty_pids.discard(pid)
            if not rows_to_write:
                return
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    """
                    INSERT INTO player_profiles (
                        player_id, hands, vpip, pfr, postflop_aggressive,
                        postflop_passive, folds_to_bet, faced_bet,
                        showdowns, showdown_bluffs, showdown_traps, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(player_id) DO UPDATE SET
                        hands=excluded.hands, vpip=excluded.vpip, pfr=excluded.pfr,
                        postflop_aggressive=excluded.postflop_aggressive,
                        postflop_passive=excluded.postflop_passive,
                        folds_to_bet=excluded.folds_to_bet, faced_bet=excluded.faced_bet,
                        showdowns=excluded.showdowns, showdown_bluffs=excluded.showdown_bluffs,
                        showdown_traps=excluded.showdown_traps, updated_at=excluded.updated_at
                    """,
                    rows_to_write,
                )
                conn.commit()
        except Exception as e:
            print(f"[SQLite] Flush warning: {e}")

    def start_new_hand(self, players):
        self.hand_vpip_flagged.clear()
        self.hand_pfr_flagged.clear()
        self.hand_aggressors.clear()
        for p in players:
            pid = p.get("id") or p.get("player_id") or p.get("name") or str(p.get("seat"))
            if pid:
                s = self._get_or_create_profile(pid)
                s["hands"] += 1
                self.dirty_pids.add(pid)

    def record_action(self, pid: str, action: str, is_preflop: bool, facing_bet: bool = False):
        if not pid:
            return
        s = self._get_or_create_profile(pid)
        self.dirty_pids.add(pid)
        if is_preflop:
            if action in ("call", "raise", "bet", "all_in") and pid not in self.hand_vpip_flagged:
                s["vpip"] += 1
                self.hand_vpip_flagged.add(pid)
            if action in ("raise", "bet", "all_in") and pid not in self.hand_pfr_flagged:
                s["pfr"] += 1
                self.hand_pfr_flagged.add(pid)
        else:
            if facing_bet:
                s["faced_bet"] += 1
                if action == "fold":
                    s["folds_to_bet"] += 1
            if action in ("raise", "bet", "all_in"):
                s["postflop_aggressive"] += 1
                self.hand_aggressors.add(pid)
            elif action in ("call", "check"):
                s["postflop_passive"] += 1

    def record_showdown_xray(self, shown_cards_dict: dict, players: list, community_cards: list):
        if not shown_cards_dict or len(community_cards) < 3:
            return
        seat_to_pid = {}
        for p in players:
            s = p.get("seat")
            pid = p.get("id") or p.get("player_id") or p.get("name") or str(s)
            if s is not None:
                seat_to_pid[str(s)] = pid
                seat_to_pid[s] = pid

        for key, cards_raw in shown_cards_dict.items():
            if not cards_raw or len(cards_raw) < 2:
                continue
            pid = seat_to_pid.get(key, str(key))
            try:
                c_ints = [Card.new(c) if isinstance(c, str) else int(c) for c in cards_raw[:2]]
                sc = self.evaluator.evaluate(community_cards, c_ints)
                rc = self.evaluator.get_rank_class(sc)
                st = self._get_or_create_profile(pid)
                st["showdowns"] += 1
                if pid in self.hand_aggressors and rc >= 8:
                    st["showdown_bluffs"] += 1
                elif pid not in self.hand_aggressors and rc <= 6:
                    st["showdown_traps"] += 1
                self.dirty_pids.add(pid)
            except Exception:
                continue

    def get_vpip(self, pid: str) -> float:
        s = self.stats.get(pid)
        if not s or s["hands"] < 6:
            return 0.30
        return max(0.10, min(0.85, s["vpip"] / s["hands"]))

    def get_aggression_factor(self, pid: str) -> float:
        s = self.stats.get(pid)
        if not s or (s["postflop_aggressive"] + s["postflop_passive"]) < 4:
            return 1.1
        return (s["postflop_aggressive"] + 1.0) / (s["postflop_passive"] + 1.0)

    def get_fold_to_bet(self, pid: str) -> float:
        s = self.stats.get(pid)
        if not s or s["faced_bet"] < 4:
            return 0.42
        return max(0.15, min(0.80, s["folds_to_bet"] / s["faced_bet"]))

    def get_bluff_frequency(self, pid: str) -> float:
        s = self.stats.get(pid)
        if not s or s["showdowns"] < 2:
            return 0.22
        return max(0.08, min(0.65, (s["showdown_bluffs"] + 0.5) / (s["showdowns"] + 2.0)))

    def build_bayesian_range_weights(self, pid: str, preflop_raised: bool, facing_3bet: bool) -> dict:
        vpip = self.get_vpip(pid) if pid else 0.30
        bluff_f = self.get_bluff_frequency(pid) if pid else 0.22
        weights = {}
        for tier_id, hands_set in RANGE_TIER.items():
            for h in hands_set:
                sc = get_preflop_score(h)
                if facing_3bet:
                    w = 1.0 if sc >= 84 else (0.35 if h in THREE_BET_BLUFFS else 0.05)
                elif preflop_raised:
                    cutoff = max(58, int(92 - vpip * 45))
                    w = 1.0 if sc >= cutoff else (0.25 if sc >= cutoff - 8 else 0.04)
                else:
                    cutoff = max(50, int(85 - vpip * 50))
                    w = 1.0 if sc >= cutoff else 0.15
                weights[h] = round(min(1.0, w + bluff_f * 0.1), 3)
        return weights


# =====================================================================
# 7. MECHANISM #9: INDEPENDENT CHIP MODEL (ICM) & BUBBLE FACTOR ENGINE
# =====================================================================
def compute_icm_bubble_factor(my_stack: int, all_stacks: list, profit_lock_stack: int = PROFIT_LOCK_STACK) -> float:
    if not all_stacks or my_stack <= 0:
        return 1.0
    total_chips = sum(max(1, s) for s in all_stacks)
    avg_stack = total_chips / max(1, len(all_stacks))

    lock_proximity = my_stack / max(1, profit_lock_stack)
    bubble_factor = 1.0
    if 0.80 <= lock_proximity < 1.0:
        bubble_factor += (lock_proximity - 0.80) * 1.4

    if TOURNAMENT_MODE and len(all_stacks) >= 3:
        larger_stacks = sum(1 for s in all_stacks if s > my_stack * 1.15)
        shorter_stacks = sum(1 for s in all_stacks if s < my_stack * 0.65)
        if larger_stacks >= 1 and shorter_stacks >= 1:
            bubble_factor += 0.22
        elif my_stack > avg_stack * 1.6:
            bubble_factor = max(0.92, bubble_factor - 0.10)

    return round(max(0.90, min(1.65, bubble_factor)), 3)


# =====================================================================
# 8. MECHANISMS #2, #5, #8: SIZING TREES, MULTI-STREET PLANNER & RNG BLEND
# =====================================================================
def empty_line_plan(hand_id=None):
    return {
        "hand_id": hand_id,
        "line": "none",
        "streets_bet": 0,
        "flop_checked": False,
        "chips_invested": 0,
        "committed": False,
        "target_spr_geometric": 0.0,
    }


class PokerEngineV5Master:
    def __init__(self, tracker: OpponentTracker, pool: ProcessPoolExecutor):
        self.evaluator = Evaluator()
        self.tracker = tracker
        self.pool = pool

    def compute_adaptive_budget(self, street: int, pot: int, to_call: int, big_blind: int):
        pot_bb = (pot + to_call) / max(1, big_blind)
        if street == 0 and to_call <= big_blind:
            return MIN_ADAPTIVE_TIME, 2.2
        if street == 0:
            return 1.2, TARGET_ACTION_TIME
        if pot_bb >= 12 or to_call >= big_blind * 4:
            return 3.0, TARGET_ACTION_TIME
        return 1.8, TARGET_ACTION_TIME

    async def estimate_equity_adaptive(
        self, hole_cards, community_cards, num_opponents: int, bayesian_weights: dict,
        postflop_aggression: int, opp_bluff_freq: float, decision_thresholds: list,
        min_time: float, max_time: float,
    ):
        if len(hole_cards) < 2:
            return 0.0, 0, False

        num_opponents = max(1, min(5, num_opponents))
        per_worker_iters = max(1000, MC_MAX_ITERS // CPU_WORKERS)
        loop = asyncio.get_running_loop()

        tasks = []
        base_seed = random.randrange(1_000_000_000)
        for w_idx in range(CPU_WORKERS):
            args = (
                hole_cards, community_cards, num_opponents, bayesian_weights,
                postflop_aggression, opp_bluff_freq, decision_thresholds,
                min_time, max_time, per_worker_iters, base_seed + w_idx * 9973,
            )
            tasks.append(loop.run_in_executor(self.pool, _adaptive_mc_worker, args))

        results = await asyncio.gather(*tasks)
        total_wins = sum(r[0] for r in results)
        total_weight = sum(r[1] for r in results)
        total_iters = sum(r[2] for r in results)
        early_stopped = all(r[3] for r in results)

        equity = (total_wins / total_weight) if total_weight > 0 else 0.5
        return equity, total_iters, early_stopped

    def build_dynamic_sizing_tree(
        self, street: int, pot: int, to_call: int, effective_stack: int,
        min_raise: int, max_raise: int, adj_equity: float, opp_fold_freq: float,
        active_opponents: int, board_info: dict, blocker_score: float,
    ):
        eff_pot = max(1, pot + to_call)
        rem_stack = max(0, effective_stack - to_call)
        spr = rem_stack / eff_pot
        streets_left = max(1, (6 - street) if street >= 3 else 3)

        geo_frac = (((2.0 * spr + 1.0) ** (1.0 / streets_left)) - 1.0) / 2.0
        geo_frac = max(0.30, min(1.50, geo_frac))

        fracs = [0.33, 0.55, geo_frac, 0.80]
        if board_info.get("wet"):
            fracs.append(0.95)
        if adj_equity >= 0.78 or blocker_score >= 0.20:
            fracs.append(1.25)

        candidates = []
        for f in sorted(set(round(x, 2) for x in fracs)):
            raw_size = int(to_call + eff_pot * f)
            clamped = max(min_raise, min(raw_size, max_raise))
            if clamped not in candidates:
                candidates.append(clamped)

        if not candidates:
            candidates = [min_raise]

        candidate_evs = []
        best_size = candidates[0]
        best_ev = -1e9

        for size in candidates:
            risk = size
            size_ratio = (size - to_call) / eff_pot
            blocker_fold_boost = blocker_score * 0.12 if size_ratio >= 0.65 else 0.0
            single_fold_p = min(0.80, (opp_fold_freq + blocker_fold_boost) * (0.72 + 0.46 * min(1.6, size_ratio)))
            if board_info.get("wet"):
                single_fold_p *= 0.86
            p_all_fold = single_fold_p ** max(1, active_opponents)

            eq_when_called = max(0.05, adj_equity - 0.06 * min(1.3, size_ratio))
            called_pot = pot + size + (size - to_call)
            ev_fold = p_all_fold * pot
            ev_called = (1.0 - p_all_fold) * (eq_when_called * called_pot - risk)
            total_ev = ev_fold + ev_called

            candidate_evs.append((size, total_ev))
            if total_ev > best_ev:
                best_ev = total_ev
                best_size = size

        return best_size, best_ev, candidate_evs

    @staticmethod
    def sample_rng_blend(prob_aggressive: float, hand_id: str, street: int) -> bool:
        if prob_aggressive >= 0.96:
            return True
        if prob_aggressive <= 0.04:
            return False
        seed_val = hash((str(hand_id), street, int(time.time() * 10) % 997))
        rng = random.Random(seed_val)
        return rng.random() < prob_aggressive

    def format_valid_action(self, desired_action: str, desired_amount: int, valid_actions_list):
        actions_map = {a["action"]: a for a in valid_actions_list}
        to_call = actions_map.get("call", {}).get("amount", 0)

        if desired_action == "raise":
            if "raise" in actions_map:
                r_info = actions_map["raise"]
                r_min = r_info.get("min", r_info.get("amount", max(20, to_call * 2)))
                r_max = r_info.get("max", r_min)
                clamped = max(r_min, min(int(desired_amount), r_max))
                return "raise", clamped
            elif "bet" in actions_map:
                b_info = actions_map["bet"]
                b_min = b_info.get("min", b_info.get("amount", 20))
                b_max = b_info.get("max", b_min)
                clamped = max(b_min, min(int(desired_amount), b_max))
                return "bet", clamped
            elif "call" in actions_map:
                return "call", to_call
            elif "check" in actions_map:
                return "check", 0

        if desired_action == "call":
            if "call" in actions_map:
                return "call", to_call
            if "check" in actions_map:
                return "check", 0

        if desired_action == "check":
            if "check" in actions_map:
                return "check", 0
            if to_call == 0 and "call" in actions_map:
                return "call", 0
            return ("fold", 0) if "fold" in actions_map else ("check", 0)

        if to_call == 0 and "check" in actions_map:
            return "check", 0
        if "fold" in actions_map:
            return "fold", 0
        first = valid_actions_list[0]
        return first["action"], first.get("amount", first.get("min", 0)) or 0

    async def decide(self, state: dict):
        hole_cards = state["hole_cards"]
        community_cards = state["community_cards"]
        pot = state["pot"]
        valid_actions = state["valid_actions"]
        position = state["position"]
        big_blind = state.get("big_blind", 20)
        active_opponents = state.get("active_opponents", 1)
        aggressor_id = state.get("aggressor_id")
        opp_af = state.get("opp_af", 1.1)
        opp_fold_freq = state.get("opp_fold_freq", 0.42)
        opp_bluff_freq = state.get("opp_bluff_freq", 0.22)
        icm_bubble_factor = state.get("icm_bubble_factor", 1.0)
        profit_lock_active = state.get("profit_lock_active", False)
        plan = state.get("line_plan") or empty_line_plan()
        state["line_plan"] = plan

        actions_map = {a["action"]: a for a in valid_actions}
        to_call = actions_map.get("call", {}).get("amount", 0)
        raise_info = actions_map.get("raise") or actions_map.get("bet") or {}
        min_raise = raise_info.get("min", big_blind * 2)
        max_raise = raise_info.get("max", state.get("stack", BUY_IN_AMOUNT))
        effective_stack = max(state.get("stack", BUY_IN_AMOUNT), max_raise, to_call)

        if len(hole_cards) < 2:
            return self.format_valid_action("check", 0, valid_actions), 0.0, 0, False

        hand_str = cards_to_hand_str(hole_cards[0], hole_cards[1])
        pf_score = get_preflop_score(hand_str)
        tier = hand_preflop_tier(hand_str)
        in_position = position in ("BTN", "CO")
        street = len(community_cards)

        raw_pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0
        pot_odds = min(0.95, raw_pot_odds * icm_bubble_factor)

        board_info = analyze_board_and_draws(hole_cards, community_cards, self.evaluator)
        blocker_info = compute_blocker_effects(hole_cards, community_cards)
        blocker_score = blocker_info["blocker_score"]

        postflop_agg = 0
        if street >= 3 and to_call > 0:
            postflop_agg = 2 if (to_call >= pot * 0.55 or to_call >= big_blind * 8) else 1

        facing_3bet = (street == 0 and to_call > big_blind * 4.5)
        bayesian_weights = self.tracker.build_bayesian_range_weights(
            aggressor_id, preflop_raised=(to_call > big_blind), facing_3bet=facing_3bet
        )

        req_call_threshold = min(0.90, pot_odds + (0.03 if in_position else 0.06))
        value_bet_threshold = 0.58 if in_position else 0.63
        jam_threshold = 0.76
        decision_thresholds = [req_call_threshold, value_bet_threshold, jam_threshold]

        min_t, max_t = self.compute_adaptive_budget(street, pot, to_call, big_blind)

        equity, rollouts, early_stopped = await self.estimate_equity_adaptive(
            hole_cards=hole_cards, community_cards=community_cards, num_opponents=active_opponents,
            bayesian_weights=bayesian_weights, postflop_aggression=postflop_agg,
            opp_bluff_freq=opp_bluff_freq, decision_thresholds=decision_thresholds,
            min_time=min_t, max_time=max_t,
        )

        draw_bonus = 0.0
        if street == 3 and board_info["draw_outs"] >= 8:
            draw_bonus = 0.03 if board_info["hero_nut_flush_draw"] else 0.015
        adj_equity = min(0.99, max(0.0, equity + draw_bonus))

        if board_info["playing_board"] and to_call > 0:
            adj_equity = min(adj_equity, 0.15)

        spr = effective_stack / max(1, pot)
        invested_ratio = plan.get("chips_invested", 0) / max(1, effective_stack + plan.get("chips_invested", 0))
        if invested_ratio >= 0.38 or (spr <= 1.6 and adj_equity >= 0.54):
            plan["committed"] = True

        def finish(desired_action, desired_amount=0):
            a, amt = self.format_valid_action(desired_action, desired_amount, valid_actions)
            if a in ("call", "raise", "bet", "all_in"):
                plan["chips_invested"] = plan.get("chips_invested", 0) + (amt if a != "call" else to_call)
            if street >= 3:
                if a in ("raise", "bet", "all_in"):
                    plan["streets_bet"] = plan.get("streets_bet", 0) + 1
                    if adj_equity >= 0.65:
                        plan["line"] = "value_3barrel"
                    elif board_info["draw_outs"] >= 8 or blocker_score >= 0.18:
                        plan["line"] = "semibluff_barrel"
                elif a == "check" and street == 3:
                    plan["flop_checked"] = True
                    if adj_equity >= 0.52:
                        plan["line"] = "delayed_cbet"
            state["line_plan"] = plan
            return (a, amt), adj_equity, rollouts, early_stopped

        if profit_lock_active and to_call > 0:
            if street == 0 and hand_str not in ("AA", "KK"):
                return finish("fold", 0)
            if street >= 3 and (board_info["rank_class"] > 5 or adj_equity < 0.82):
                return finish("fold", 0)

        if street == 0:
            stack_bb = effective_stack / big_blind if big_blind > 0 else 75
            is_unopened = to_call <= big_blind

            if to_call == 0:
                return finish("check", 0)

            if stack_bb <= 12:
                if pf_score >= 78 or (pf_score >= 70 and position in ("BTN", "SB", "CO")):
                    return finish("raise", max_raise)
                return finish("check" if to_call == 0 else "fold", 0)

            p_raise, p_call, _ = get_gto_preflop_frequencies(
                position, hand_str, facing_open=not is_unopened, in_position=in_position
            )

            if is_unopened:
                open_size = int(big_blind * 2.5) if position != "SB" else int(big_blind * 3.0)
                if self.sample_rng_blend(p_raise, plan.get("hand_id"), street):
                    return finish("raise", open_size)
                if position == "BB":
                    return finish("check" if to_call == 0 else "call", to_call)
                if position == "SB" and (pf_score >= 48 or hand_str.endswith("s")):
                    return finish("call", to_call)
                return finish("check" if to_call == 0 else "fold", 0)

            if to_call > big_blind * 9:
                if hand_str in ("AA", "KK"):
                    return finish("raise", max_raise)
                if hand_str in ("QQ", "AKs") and to_call <= effective_stack * 0.45 and icm_bubble_factor <= 1.15:
                    return finish("call", to_call)
                return finish("fold", 0)

            if to_call > big_blind * 4.5:
                if hand_str in FOUR_BET_VALUE:
                    return finish("raise", int(to_call * 2.3))
                if pf_score >= 85 and (in_position or to_call <= effective_stack * 0.18):
                    return finish("call", to_call)
                return finish("fold", 0)

            three_bet_size = int(to_call * (3.0 if in_position else 3.5))
            if tier == 1 or (pf_score >= 85 and in_position and active_opponents <= 2):
                return finish("raise", three_bet_size)
            if self.sample_rng_blend(p_raise, plan.get("hand_id"), street) and active_opponents <= 2:
                return finish("raise", three_bet_size)

            if position == "BB":
                is_pocket_pair = hand_str[0] == hand_str[1]
                if (pf_score >= 58 or is_pocket_pair) and to_call <= effective_stack * 0.10:
                    return finish("call", to_call)

            if position in ("BTN", "CO", "SB") and hand_str.endswith("s") and pf_score >= 60:
                if to_call <= effective_stack * 0.05 and to_call <= big_blind * 2.5:
                    return finish("call", to_call)

            if hand_str[0] == hand_str[1] and effective_stack >= to_call * 12:
                return finish("call", to_call)
            if in_position and pf_score >= 70 and adj_equity >= pot_odds:
                return finish("call", to_call)
            if pf_score >= 77 and adj_equity >= (pot_odds + 0.02):
                return finish("call", to_call)

            return finish("fold", 0)

        rc = board_info["rank_class"]
        is_monster = rc <= 5 and not board_info["playing_board"]
        is_trips_or_two_pair = (
            rc in (6, 7) and not board_info["vulnerable_one_pair"] and not board_info["playing_board"]
        )
        danger_board = (board_info["four_flush"] and rc > 4) or (board_info["four_straight"] and rc > 5)

        best_sz, best_ev, candidate_evs = self.build_dynamic_sizing_tree(
            street, pot, to_call, effective_stack, min_raise, max_raise,
            adj_equity, opp_fold_freq, active_opponents, board_info, blocker_score,
        )
        cfr_mix = LightweightSubgameCFR.solve_postflop_mix(
            adj_equity, pot, to_call, candidate_evs, blocker_score
        )
        cfr_agg_prob = sum(cfr_mix[2:])

        if to_call == 0:
            if danger_board and not is_monster and blocker_score < 0.20:
                return finish("check", 0)

            if is_monster or (is_trips_or_two_pair and adj_equity >= 0.68) or plan.get("committed"):
                return finish("raise", best_sz)

            if street == 4 and plan.get("line") == "delayed_cbet" and adj_equity >= 0.50:
                return finish("raise", best_sz)

            if adj_equity >= value_bet_threshold:
                if street == 5 and board_info["vulnerable_one_pair"] and board_info["wet"]:
                    return finish("check", 0)
                if best_ev > adj_equity * pot or self.sample_rng_blend(cfr_agg_prob, plan.get("hand_id"), street):
                    return finish("raise", best_sz)
                return finish("raise", int(max(min_raise, pot * 0.42)))

            if (
                street < 5 and active_opponents == 1 and (board_info["draw_outs"] >= 8 or blocker_score >= 0.22)
                and (in_position or plan.get("streets_bet", 0) <= 1)
            ):
                return finish("raise", int(max(min_raise, pot * 0.48)))

            if (
                street == 5 and active_opponents == 1 and blocker_info["blocks_nut_flush"]
                and opp_fold_freq >= 0.45 and self.sample_rng_blend(0.35, plan.get("hand_id"), street)
            ):
                return finish("raise", best_sz)

            if (
                street == 3 and active_opponents == 1 and in_position and board_info["dry"]
                and adj_equity >= 0.36 and opp_fold_freq >= 0.38
            ):
                return finish("raise", int(max(min_raise, pot * 0.33)))

            return finish("check", 0)

        if board_info["playing_board"]:
            return finish("fold", 0)
        if danger_board and not is_monster and to_call > pot * 0.25 and not blocker_info["blocks_nut_flush"]:
            return finish("fold", 0)

        if board_info["vulnerable_one_pair"] or rc >= 8:
            bluffer_bonus = 0.12 if opp_bluff_freq >= 0.35 else 0.0
            max_one_pair_call = BUY_IN_AMOUNT * ((0.35 if opp_af > 2.2 else 0.22) + bluffer_bonus)
            if street >= 4 and to_call > max_one_pair_call and adj_equity < (0.76 - bluffer_bonus):
                return finish("fold", 0)
            if street == 5 and to_call > pot * 0.65 and adj_equity < (0.68 - bluffer_bonus) and opp_af < 1.8:
                return finish("fold", 0)

        street_margin = 0.03 if street == 3 else (0.05 if street == 4 else 0.07)
        if board_info["wet"]:
            street_margin += 0.02
        pos_margin = 0.0 if in_position else 0.03
        nit_margin = 0.04 if (opp_af < 0.9 and to_call > pot * 0.5) else 0.0
        blocker_discount = 0.03 if (blocker_info["blocks_top_pair"] or blocker_info["unblocks_folding_range"]) else 0.0
        required_equity = pot_odds + street_margin + pos_margin + nit_margin - blocker_discount

        if (is_monster and adj_equity >= 0.76) or (is_trips_or_two_pair and adj_equity >= 0.80):
            raise_target = max_raise if spr <= 2.2 else best_sz
            return finish("raise", raise_target)

        if plan.get("committed") and adj_equity >= max(0.38, pot_odds - 0.05):
            return finish("call", to_call)

        if adj_equity >= required_equity:
            return finish("call", to_call)

        if adj_equity >= (required_equity - 0.02) and self.sample_rng_blend(0.45, plan.get("hand_id"), street):
            return finish("call", to_call)

        if street < 5 and board_info["draw_outs"] >= 8:
            implied_discount = 0.06 if (in_position and spr >= 3.5) else 0.03
            if adj_equity >= (pot_odds - implied_discount) and to_call <= pot * 0.65:
                return finish("call", to_call)

        return finish("fold", 0)


# =====================================================================
# 9. WEBSOCKET CLIENT WITH 5K RATHOLING, SHOWDOWN X-RAY & RESYNC
# =====================================================================
class OpenPokerClientV5:
    def __init__(self, api_key: str, ws_url: str, pool: ProcessPoolExecutor):
        self.api_key = api_key
        self.ws_url = ws_url
        self.tracker = OpponentTracker(DB_PATH)
        self.engine = PokerEngineV5Master(self.tracker, pool)

        self.bot_name = None
        self.my_player_id = None
        self.my_seat = None
        self.my_stack = BUY_IN_AMOUNT
        self.pending_rathole = False

        self.hole_cards = []
        self.community_cards = []
        self.dealer_seat = 0
        self.table_id = None
        self.last_table_seq = 0
        self.players = []
        self.folded_seats = set()
        self.last_aggressor_id = None
        self.current_street_bet = False
        self.line_plan = empty_line_plan()

        self.hands_played = 0
        self.hands_won = 0
        self.ratholes_completed = 0
        self.last_action_summary = "Waiting for first hand..."
        self.connected = False
        self.boot_time = time.time()

    def check_active_game(self):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = requests.get(f"{REST_BASE_URL}/api/me/active-game", headers=headers, timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"[REST] Active-game check skipped: {e}")
        return None

    def update_hero_info_from_players(self, players):
        if not players:
            return
        self.players = players
        for p in players:
            pid = p.get("id") or p.get("player_id")
            pname = p.get("name")
            pseat = p.get("seat")
            if (
                (self.my_seat is not None and pseat == self.my_seat)
                or (self.my_player_id and pid == self.my_player_id)
                or (self.bot_name and pname == self.bot_name)
            ):
                if pseat is not None:
                    self.my_seat = pseat
                if pid:
                    self.my_player_id = pid

                for key in ("stack", "chips", "new_stack", "amount", "balance"):
                    if key in p and p[key] is not None:
                        val = int(p[key])
                        if val > 0:
                            self.my_stack = val
                            break

                if self.my_stack >= PROFIT_LOCK_STACK:
                    self.pending_rathole = True
                break

    async def execute_rathole_leave_and_rejoin(self, ws):
        print(
            f"[Profit-Lock] Stack reached {self.my_stack} (>= {PROFIT_LOCK_STACK})! "
            f"Leaving table to lock profit and rejoining with {BUY_IN_AMOUNT}..."
        )
        self.pending_rathole = False
        self.ratholes_completed += 1
        await ws.send(json.dumps({"type": "leave_table"}))
        await asyncio.sleep(1.5)
        self.my_stack = BUY_IN_AMOUNT
        self.table_id = None
        await ws.send(json.dumps({"type": "join_lobby", "buy_in": BUY_IN_AMOUNT}))

    def get_my_position(self, my_seat, players) -> str:
        if not players or my_seat is None:
            return "MP"
        active_seats = sorted(
            [p["seat"] for p in players if "seat" in p and p.get("stack", p.get("chips", 1)) > 0]
        )
        if my_seat not in active_seats:
            active_seats = sorted([p["seat"] for p in players if "seat" in p])
        if my_seat not in active_seats or not active_seats:
            return "MP"

        dealer_idx = 0
        for i, s in enumerate(active_seats):
            if s <= (self.dealer_seat or 0):
                dealer_idx = i
        rotated = active_seats[dealer_idx:] + active_seats[:dealer_idx]
        rel = rotated.index(my_seat)
        n = len(active_seats)
        if n == 2:
            return "BTN" if rel == 0 else "BB"
        if rel == 0:
            return "BTN"
        if rel == 1:
            return "SB"
        if rel == 2:
            return "BB"
        if rel == n - 1:
            return "CO"
        if rel == 3 and n >= 5:
            return "UTG"
        return "MP"

    async def handle_your_turn(self, ws, msg: dict):
        turn_token = msg.get("turn_token")
        if not turn_token:
            return

        hand_id = msg.get("hand_id")
        pot = msg.get("pot", 0)
        valid_actions = msg.get("valid_actions", [])
        if not valid_actions:
            return

        if "seat" in msg and msg["seat"] is not None:
            self.my_seat = msg["seat"]
        if "players" in msg and msg["players"]:
            self.update_hero_info_from_players(msg["players"])
        if "community_cards" in msg and msg["community_cards"] is not None:
            self.community_cards = [Card.new(c) for c in msg["community_cards"]]
        if "hole_cards" in msg and msg["hole_cards"]:
            self.hole_cards = [Card.new(c) for c in msg["hole_cards"]]
        if "dealer_seat" in msg or "button_seat" in msg:
            self.dealer_seat = msg.get("dealer_seat", msg.get("button_seat", self.dealer_seat))

        for key in ("stack", "chips", "max_raise", "balance"):
            if key in msg and msg[key] is not None:
                val = int(msg[key])
                if val > self.my_stack:
                    self.my_stack = val

        for va in valid_actions:
            if va.get("action") in ("raise", "bet") and va.get("max"):
                val = int(va["max"])
                if val > self.my_stack:
                    self.my_stack = val
            if va.get("stack"):
                val = int(va["stack"])
                if val > self.my_stack:
                    self.my_stack = val

        if self.my_stack >= PROFIT_LOCK_STACK:
            self.pending_rathole = True

        position = self.get_my_position(self.my_seat, self.players)

        if self.line_plan.get("hand_id") != hand_id:
            self.line_plan = empty_line_plan(hand_id)

        total_players = len(self.players) if self.players else 6
        active_opponents = max(1, total_players - len(self.folded_seats) - 1)

        opp_af = (
            self.tracker.get_aggression_factor(self.last_aggressor_id)
            if self.last_aggressor_id
            else 1.1
        )
        opp_fold_freq = (
            self.tracker.get_fold_to_bet(self.last_aggressor_id)
            if self.last_aggressor_id
            else 0.42
        )
        opp_bluff_freq = (
            self.tracker.get_bluff_frequency(self.last_aggressor_id)
            if self.last_aggressor_id
            else 0.22
        )

        all_stacks = [
            int(p.get("stack", p.get("chips", BUY_IN_AMOUNT)))
            for p in self.players
            if isinstance(p, dict)
        ] or [self.my_stack]
        icm_bf = compute_icm_bubble_factor(self.my_stack, all_stacks, PROFIT_LOCK_STACK)

        state = {
            "hole_cards": self.hole_cards,
            "community_cards": self.community_cards,
            "pot": pot,
            "valid_actions": valid_actions,
            "position": position,
            "big_blind": msg.get("big_blind", 20),
            "stack": self.my_stack,
            "active_opponents": active_opponents,
            "aggressor_id": self.last_aggressor_id,
            "opp_af": opp_af,
            "opp_fold_freq": opp_fold_freq,
            "opp_bluff_freq": opp_bluff_freq,
            "icm_bubble_factor": icm_bf,
            "profit_lock_active": self.pending_rathole,
            "line_plan": self.line_plan,
        }

        t0 = time.perf_counter()
        (action, amount), equity, rollouts, early_stopped = await self.engine.decide(state)
        self.line_plan = state.get("line_plan", self.line_plan)
        elapsed = time.perf_counter() - t0

        payload = {
            "type": "action",
            "hand_id": hand_id,
            "client_action_id": str(uuid.uuid4()),
            "turn_token": turn_token,
            "action": action,
        }
        if action in ("raise", "bet", "call") and amount and amount > 0:
            payload["amount"] = int(amount)

        await ws.send(json.dumps(payload))

        hc_str = " ".join([Card.int_to_str(c) for c in self.hole_cards]) if self.hole_cards else "??"
        bd_str = (
            " ".join([Card.int_to_str(c) for c in self.community_cards])
            if self.community_cards
            else "Preflop"
        )
        mode_tag = "CONV-EARLY" if early_stopped else "FULL-DEEP"
        win_rate_pct = (self.hands_won / self.hands_played * 100) if self.hands_played else 0.0
        self.last_action_summary = (
            f"[{position} | Stack: {self.my_stack} | ICM-BF: {icm_bf:.2f} | WR: {win_rate_pct:.1f}%] "
            f"Cards: {hc_str} | Board: {bd_str} | "
            f"Eq: {equity*100:.1f}% ({rollouts:,} sims [{mode_tag}] in {elapsed:.2f}s) "
            f"-> {action.upper()} {amount}"
        )
        print(self.last_action_summary)

    async def run_forever(self):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        header_kwarg = (
            "additional_headers"
            if "additional_headers" in inspect.signature(websockets.connect).parameters
            else "extra_headers"
        )
        backoff = 2

        while True:
            try:
                active = await asyncio.to_thread(self.check_active_game)
                print(f"[Client] Connecting to {self.ws_url}...")

                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                    **{header_kwarg: headers},
                ) as ws:
                    self.connected = True
                    backoff = 2
                    print("[WebSocket] Connected to OpenPoker.")

                    await ws.send(json.dumps({"type": "set_auto_rebuy", "enabled": True}))

                    if active and active.get("playing") and active.get("table_id"):
                        self.table_id = active["table_id"]
                        print(f"[Resync] Resuming active table {self.table_id}...")
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "resync_request",
                                    "table_id": self.table_id,
                                    "last_table_seq": self.last_table_seq,
                                }
                            )
                        )
                    else:
                        await ws.send(json.dumps({"type": "join_lobby", "buy_in": BUY_IN_AMOUNT}))

                    async for raw_message in ws:
                        msg = json.loads(raw_message)
                        msg_type = msg.get("type")

                        if "table_seq" in msg:
                            self.last_table_seq = max(self.last_table_seq, msg["table_seq"])
                        if "table_id" in msg:
                            self.table_id = msg["table_id"]

                        if msg_type == "connected":
                            self.bot_name = msg.get("name")
                            self.my_player_id = msg.get("player_id") or msg.get("id")
                            print(f"[Server] Authenticated as: {self.bot_name or 'Bot'}")

                        elif msg_type == "error":
                            code = msg.get("code")
                            print(f"[Server Error] {code}: {msg.get('message')}")
                            if code == "already_seated":
                                if self.pending_rathole:
                                    await self.execute_rathole_leave_and_rejoin(ws)
                                elif msg.get("table_id"):
                                    self.table_id = msg["table_id"]
                                    await ws.send(
                                        json.dumps(
                                            {
                                                "type": "resync_request",
                                                "table_id": self.table_id,
                                                "last_table_seq": 0,
                                            }
                                        )
                                    )
                            elif code == "insufficient_season_chips":
                                print("[Bankroll] Low season chips. Triggering rebuy...")
                                await ws.send(json.dumps({"type": "rebuy", "amount": 0}))
                                await asyncio.sleep(2)
                                await ws.send(json.dumps({"type": "join_lobby", "buy_in": 1000}))

                        elif msg_type in ("hand_start", "new_hand"):
                            self.hands_played += 1
                            self.hole_cards = []
                            self.community_cards = []
                            self.folded_seats.clear()
                            self.last_aggressor_id = None
                            self.current_street_bet = False
                            self.line_plan = empty_line_plan(msg.get("hand_id"))
                            if "seat" in msg or "your_seat" in msg:
                                self.my_seat = msg.get("seat", msg.get("your_seat", self.my_seat))
                            self.dealer_seat = msg.get(
                                "dealer_seat", msg.get("button_seat", msg.get("dealer", 0))
                            )
                            if "players" in msg:
                                self.update_hero_info_from_players(msg["players"])
                            self.tracker.start_new_hand(self.players)

                        elif msg_type == "hole_cards":
                            self.hole_cards = [Card.new(c) for c in msg.get("cards", [])]

                        elif msg_type == "community_cards":
                            self.community_cards = [Card.new(c) for c in msg.get("cards", [])]
                            self.current_street_bet = False

                        elif msg_type == "player_action":
                            seat = msg.get("seat")
                            act = msg.get("action")
                            pid = msg.get("player_id") or msg.get("name") or str(seat)
                            is_pf = len(self.community_cards) == 0
                            if act == "fold" and seat is not None:
                                self.folded_seats.add(seat)
                            self.tracker.record_action(
                                pid=pid,
                                action=act,
                                is_preflop=is_pf,
                                facing_bet=self.current_street_bet,
                            )
                            if act in ("raise", "bet", "all_in"):
                                self.last_aggressor_id = pid
                                self.current_street_bet = True

                        elif msg_type in ("hand_result", "hand_end", "hand_complete"):
                            winners = msg.get("winners", [])
                            for w in winners:
                                w_seat = w.get("seat")
                                w_name = w.get("name") or w.get("player_id")
                                if (self.my_seat is not None and w_seat == self.my_seat) or (
                                    self.bot_name and w_name == self.bot_name
                                ):
                                    self.hands_won += 1

                            shown_raw = msg.get("shown_cards") or {}
                            if isinstance(shown_raw, str):
                                try:
                                    shown_raw = json.loads(shown_raw)
                                except Exception:
                                    shown_raw = {}
                            if shown_raw:
                                self.tracker.record_showdown_xray(
                                    shown_raw, self.players, self.community_cards
                                )

                            if self.hands_played % 50 == 0:
                                asyncio.create_task(asyncio.to_thread(self.tracker.flush_to_sqlite))
                                trim_process_memory()

                            if "players" in msg:
                                self.update_hero_info_from_players(msg["players"])

                            for key in ("stack", "new_stack", "chips", "balance"):
                                if key in msg and msg[key] is not None:
                                    self.my_stack = int(msg[key])

                            if self.my_stack >= PROFIT_LOCK_STACK or self.pending_rathole:
                                await self.execute_rathole_leave_and_rejoin(ws)

                        elif msg_type == "resync_response":
                            snap = msg.get("snapshot", {})
                            if snap.get("community_cards"):
                                self.community_cards = [
                                    Card.new(c) for c in snap["community_cards"]
                                ]
                            if snap.get("players"):
                                self.update_hero_info_from_players(snap["players"])
                            hero = snap.get("hero", {})
                            if hero.get("seat") is not None:
                                self.my_seat = hero["seat"]
                            if hero.get("stack") is not None:
                                self.my_stack = int(hero["stack"])
                                if self.my_stack >= PROFIT_LOCK_STACK:
                                    self.pending_rathole = True
                            if hero.get("hole_cards"):
                                self.hole_cards = [Card.new(c) for c in hero["hole_cards"]]
                            if hero.get("turn_token") and hero.get("valid_actions"):
                                turn_msg = {
                                    "turn_token": hero["turn_token"],
                                    "valid_actions": hero["valid_actions"],
                                    "hand_id": snap.get("hand_id"),
                                    "pot": snap.get("pot", 0),
                                    "seat": hero.get("seat"),
                                    "players": snap.get("players", []),
                                }
                                await self.handle_your_turn(ws, turn_msg)

                        elif msg_type == "your_turn":
                            await self.handle_your_turn(ws, msg)

                        elif msg_type == "action_rejected":
                            print(
                                f"[Warning] Action rejected: {msg.get('reason')} "
                                f"({msg.get('attempted_action')})"
                            )

                        elif msg_type == "auto_rebuy_scheduled":
                            wait_sec = msg.get("cooldown_seconds", 300)
                            print(f"[Rebuy] Cooldown active. Waiting {wait_sec}s...")
                            await asyncio.sleep(wait_sec + 2)
                            self.my_stack = BUY_IN_AMOUNT
                            await ws.send(
                                json.dumps({"type": "join_lobby", "buy_in": BUY_IN_AMOUNT})
                            )

                        elif msg_type == "rebuy_success":
                            print(f"[Rebuy] Chips credited! Joining with {BUY_IN_AMOUNT}...")
                            self.my_stack = BUY_IN_AMOUNT
                            await ws.send(
                                json.dumps({"type": "join_lobby", "buy_in": BUY_IN_AMOUNT})
                            )

                        elif msg_type == "busted":
                            print(f"[Table] Busted! Rejoining in 3s with {BUY_IN_AMOUNT}...")
                            self.pending_rathole = False
                            self.my_stack = BUY_IN_AMOUNT
                            await asyncio.sleep(3)
                            await ws.send(
                                json.dumps({"type": "join_lobby", "buy_in": BUY_IN_AMOUNT})
                            )

            except asyncio.CancelledError:
                print("[Shutdown] WebSocket loop cancelled cleanly.")
                self.connected = False
                raise
            except Exception as e:
                self.connected = False
                jitter = random.uniform(0.5, 2.0)
                wait_time = min(60, backoff + jitter)
                print(f"[Disconnected] {e}. Reconnecting in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                backoff = min(60, backoff * 2)


# =====================================================================
# 10. RENDER KEEP-ALIVE WEB SERVER & MAIN ENTRYPOINT
# =====================================================================
async def start_render_server(bot: OpenPokerClientV5):
    app = web.Application()

    async def health_handler(request):
        uptime_min = int((time.time() - bot.boot_time) / 60)
        win_rate = (bot.hands_won / bot.hands_played * 100) if bot.hands_played else 0.0
        status = {
            "version": "v5.1-gh-actions-4core",
            "status": "online" if bot.connected else "reconnecting",
            "uptime_minutes": uptime_min,
            "current_stack": bot.my_stack,
            "profit_lock_target": PROFIT_LOCK_STACK,
            "ratholes_completed": bot.ratholes_completed,
            "hands_played": bot.hands_played,
            "hands_won": bot.hands_won,
            "win_rate_pct": round(win_rate, 2),
            "last_decision": bot.last_action_summary,
            "tracked_opponents": len(bot.tracker.stats),
            "sqlite_db": DB_PATH,
        }
        return web.json_response(status)

    app.router.add_get("/", health_handler)
    app.router.add_get("/health", health_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    print(f"[Server] Health dashboard listening on http://0.0.0.0:{PORT}")


async def self_ping_loop():
    if os.getenv("GITHUB_ACTIONS") == "true":
        return
        
    await asyncio.sleep(15)
    target_url = (
        f"{RENDER_EXTERNAL_URL.rstrip('/')}/health"
        if RENDER_EXTERNAL_URL
        else f"http://127.0.0.1:{PORT}/health"
    )
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(target_url, timeout=10) as resp:
                    print(f"[Keep-Alive] Pinged {target_url} -> HTTP {resp.status}")
            except Exception as e:
                print(f"[Keep-Alive] Ping warning: {e}")
            await asyncio.sleep(600)


async def main():
    pool_kwargs = {"max_workers": CPU_WORKERS, "initializer": _init_worker}
    if "max_tasks_per_child" in inspect.signature(ProcessPoolExecutor).parameters:
        pool_kwargs["max_tasks_per_child"] = 200
    pool = ProcessPoolExecutor(**pool_kwargs)
    
    bot = OpenPokerClientV5(api_key=API_KEY, ws_url=WS_URL, pool=pool)
    await start_render_server(bot)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            signal.signal(sig, lambda s, f: loop.call_soon_threadsafe(stop_event.set))

    bot_task = asyncio.create_task(bot.run_forever())
    ping_task = asyncio.create_task(self_ping_loop())

    print(f"[Main] Bot started with {CPU_WORKERS} CPU workers. Waiting for shutdown...")
    
    # This loop ignores crashes. If a task dies, it restarts it. 
    # It ONLY exits when GitHub sends a SIGTERM (at 5h 45m).
    while not stop_event.is_set():
        await asyncio.sleep(1)
        
        if bot_task.done():
            print("[Fatal] Bot task crashed! Restarting in 5 seconds...")
            if bot_task.exception():
                print(f"Reason: {bot_task.exception()}")
            await asyncio.sleep(5)
            bot_task = asyncio.create_task(bot.run_forever())
            
        if ping_task.done():
            print("[Fatal] Ping task crashed! Restarting...")
            if ping_task.exception():
                print(f"Reason: {ping_task.exception()}")
            ping_task = asyncio.create_task(self_ping_loop())

    print("\n[Shutdown] SIGTERM or Ctrl+C received. Cleaning up...")
    
    bot_task.cancel()
    ping_task.cancel()
    await asyncio.gather(bot_task, ping_task, return_exceptions=True)

    bot.tracker.flush_to_sqlite()
    pool.shutdown(wait=False, cancel_futures=True)
    print("[Shutdown] Clean exit complete.")
