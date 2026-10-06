import streamlit as st
import cloudpickle
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="2026 College Football Predictor",
    page_icon="🏈",
    layout="centered",
)

BUNDLE_PATH = Path(__file__).parent / "model_bundle_clean.pkl"


@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            "model_bundle_clean.pkl is missing. Place it beside app_clean.py."
        )
    with open(BUNDLE_PATH, "rb") as f:
        return cloudpickle.load(f)


def safe_number(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def edge_grade(edge):
    edge = abs(float(edge))
    if edge < 1.0:
        return "PASS / market agreement"
    if edge < 2.0:
        return "SMALL LEAN"
    if edge < 3.5:
        return "LEAN"
    if edge < 5.0:
        return "STRONG LEAN"
    return "MAJOR MODEL DISAGREEMENT"


# ============================================================
# LOAD CLEAN BUNDLE
# ============================================================

try:
    bundle = load_bundle()
except Exception as exc:
    st.error(str(exc))
    st.stop()

margin_model = bundle["margin_model"]
features = list(bundle["features"])
CURRENT_PROFILES = bundle["CURRENT_PROFILES"].copy()
CURRENT_OPPONENTS = bundle["CURRENT_OPPONENTS"]
TEAM_SPLIT_PROFILE_36C12 = bundle["TEAM_SPLIT_PROFILE_36C12"].copy()
team_list = [str(x).strip() for x in bundle["team_list"]]


# ============================================================
# VALIDATE / NORMALIZE EXPORTED 2026 DATA
# ============================================================

if not isinstance(CURRENT_PROFILES, pd.DataFrame):
    raise RuntimeError(
        f"CURRENT_PROFILES should be a DataFrame, "
        f"got {type(CURRENT_PROFILES).__name__}."
    )

if "team" not in CURRENT_PROFILES.columns:
    raise RuntimeError(
        "CURRENT_PROFILES is missing the required 'team' column."
    )

CURRENT_PROFILES["team"] = (
    CURRENT_PROFILES["team"]
    .astype(str)
    .str.strip()
)

if CURRENT_PROFILES["team"].duplicated().any():
    duplicates = CURRENT_PROFILES.loc[
        CURRENT_PROFILES["team"].duplicated(keep=False),
        "team",
    ].tolist()

    raise RuntimeError(
        f"CURRENT_PROFILES contains duplicate team names: "
        f"{duplicates[:10]}"
    )

CURRENT_PROFILE_LOOKUP = CURRENT_PROFILES.set_index(
    "team",
    drop=False,
)


# ============================================================
# NORMALIZE SPLIT PROFILE TABLE
# ============================================================

if not isinstance(TEAM_SPLIT_PROFILE_36C12, pd.DataFrame):
    raise RuntimeError(
        "TEAM_SPLIT_PROFILE_36C12 should be a pandas DataFrame."
    )

if "team" in TEAM_SPLIT_PROFILE_36C12.columns:

    TEAM_SPLIT_PROFILE_36C12["team"] = (
        TEAM_SPLIT_PROFILE_36C12["team"]
        .astype(str)
        .str.strip()
    )

    if TEAM_SPLIT_PROFILE_36C12["team"].duplicated().any():
        raise RuntimeError(
            "TEAM_SPLIT_PROFILE_36C12 contains duplicate team names."
        )

    TEAM_SPLIT_LOOKUP = (
        TEAM_SPLIT_PROFILE_36C12
        .set_index("team", drop=False)
    )

else:

    TEAM_SPLIT_LOOKUP = TEAM_SPLIT_PROFILE_36C12.copy()

    TEAM_SPLIT_LOOKUP.index = (
        TEAM_SPLIT_LOOKUP.index
        .astype(str)
        .str.strip()
    )

    if TEAM_SPLIT_LOOKUP.index.duplicated().any():
        raise RuntimeError(
            "TEAM_SPLIT_PROFILE_36C12 contains duplicate team names."
        )


required_profile_fields = [
    "win_pct",
    "ppg",
    "papg",
    "avg_margin",
    "off_ppa",
    "def_ppa",
    "off_success",
    "def_success",
    "off_explosive",
    "def_explosive",
    "off_ypp",
    "def_ypp",
]

missing_profile_fields = [
    c
    for c in required_profile_fields
    if c not in CURRENT_PROFILES.columns
]

if missing_profile_fields:
    raise RuntimeError(
        f"CURRENT_PROFILES is missing required fields: "
        f"{missing_profile_fields}"
    )


required_split_fields = [
    "current_pass_ppa",
    "current_rush_ppa",
    "current_def_pass_ppa",
    "current_def_rush_ppa",
]

missing_split_fields = [
    c
    for c in required_split_fields
    if c not in TEAM_SPLIT_LOOKUP.columns
]

if missing_split_fields:
    raise RuntimeError(
        f"TEAM_SPLIT_PROFILE_36C12 is missing fields: "
        f"{missing_split_fields}"
    )


# ============================================================
# PROFILE HELPERS
# ============================================================

def get_profile(team):
    """
    Return exactly one current 2026 profile row.
    """

    team = str(team).strip()

    if team not in CURRENT_PROFILE_LOOKUP.index:
        raise ValueError(
            f"No current 2026 profile found for {team}."
        )

    row = CURRENT_PROFILE_LOOKUP.loc[team]

    if isinstance(row, pd.DataFrame):
        raise RuntimeError(
            f"Multiple current profiles found for {team}."
        )

    return row


def get_split_profile(team):
    """
    Return exactly one current pass/rush split profile.
    """

    team = str(team).strip()

    if team not in TEAM_SPLIT_LOOKUP.index:
        raise ValueError(
            f"No current split profile found for {team}."
        )

    row = TEAM_SPLIT_LOOKUP.loc[team]

    if isinstance(row, pd.DataFrame):
        raise RuntimeError(
            f"Multiple split profiles found for {team}."
        )

    return row


# ============================================================
# OPPONENT ADJUSTMENTS
# ============================================================

def opponent_def_strength(team):

    values = []

    for opponent in CURRENT_OPPONENTS.get(team, []):

        opponent = str(opponent).strip()

        if opponent not in CURRENT_PROFILE_LOOKUP.index:
            continue

        value = safe_number(
            get_profile(opponent)["def_ppa"]
        )

        if value is not None:
            values.append(value)

    if not values:
        return np.nan

    return float(np.mean(values))


def opponent_off_strength(team):

    values = []

    for opponent in CURRENT_OPPONENTS.get(team, []):

        opponent = str(opponent).strip()

        if opponent not in CURRENT_PROFILE_LOOKUP.index:
            continue

        value = safe_number(
            get_profile(opponent)["off_ppa"]
        )

        if value is not None:
            values.append(value)

    if not values:
        return np.nan

    return float(np.mean(values))


def adjusted_profile(team):

    raw = get_profile(team)

    opp_def = opponent_def_strength(team)
    opp_off = opponent_off_strength(team)

    raw_off = float(raw["off_ppa"])
    raw_def = float(raw["def_ppa"])

    if np.isfinite(opp_def):
        adj_off = raw_off - opp_def
    else:
        adj_off = raw_off

    if np.isfinite(opp_off):
        adj_def = raw_def - opp_off
    else:
        adj_def = raw_def

    return {
        "adj_off_ppa": float(adj_off),
        "adj_def_ppa": float(adj_def),
        "opp_def_strength": opp_def,
        "opp_off_strength": opp_off,
    }


# ============================================================
# CURRENT PASS / RUSH SPLIT FEATURES
# ============================================================

def calculate_live_split_features(
    away_team,
    home_team,
):

    away_profile = get_split_profile(away_team)
    home_profile = get_split_profile(home_team)

    # Home offense vs away defense

    home_pass_edge = (
        float(home_profile["current_pass_ppa"])
        -
        float(away_profile["current_def_pass_ppa"])
    )

    home_rush_edge = (
        float(home_profile["current_rush_ppa"])
        -
        float(away_profile["current_def_rush_ppa"])
    )

    # Away offense vs home defense

    away_pass_edge = (
        float(away_profile["current_pass_ppa"])
        -
        float(home_profile["current_def_pass_ppa"])
    )

    away_rush_edge = (
        float(away_profile["current_rush_ppa"])
        -
        float(home_profile["current_def_rush_ppa"])
    )

    # Historical model definition:
    # home matchup edge - away matchup edge

    pass_matchup_diff = (
        home_pass_edge
        -
        away_pass_edge
    )

    rush_matchup_diff = (
        home_rush_edge
        -
        away_rush_edge
    )

    if not np.isfinite(pass_matchup_diff):
        raise RuntimeError(
            f"pass_matchup_diff is non-finite for "
            f"{away_team} @ {home_team}."
        )

    if not np.isfinite(rush_matchup_diff):
        raise RuntimeError(
            f"rush_matchup_diff is non-finite for "
            f"{away_team} @ {home_team}."
        )

    return {
        "pass_matchup_diff":
            float(pass_matchup_diff),

        "rush_matchup_diff":
            float(rush_matchup_diff),
    }


# ============================================================
# CLEAN 21-FEATURE BUILDER
# ============================================================

def build_live_game_features(
    away_team,
    home_team,
    neutral_site=False,
):

    away_team = str(away_team).strip()
    home_team = str(home_team).strip()

    if away_team == home_team:
        raise ValueError(
            "Home and away teams cannot be the same."
        )

    away = get_profile(away_team)
    home = get_profile(home_team)

    away_adj = adjusted_profile(away_team)
    home_adj = adjusted_profile(home_team)


    # --------------------------------------------------------
    # BASIC DIFFERENCES
    # --------------------------------------------------------

    win_pct_diff = (
        float(home["win_pct"])
        -
        float(away["win_pct"])
    )

    avg_margin_diff = (
        float(home["avg_margin"])
        -
        float(away["avg_margin"])
    )


    # --------------------------------------------------------
    # SCORING MATCHUP
    # --------------------------------------------------------

    home_scoring_matchup = (
        float(home["ppg"])
        -
        float(away["papg"])
    )

    away_scoring_matchup = (
        float(away["ppg"])
        -
        float(home["papg"])
    )

    scoring_matchup_diff = (
        home_scoring_matchup
        -
        away_scoring_matchup
    )


    # --------------------------------------------------------
    # PPA MATCHUP
    # --------------------------------------------------------

    home_ppa_matchup = (
        float(home["off_ppa"])
        -
        float(away["def_ppa"])
    )

    away_ppa_matchup = (
        float(away["off_ppa"])
        -
        float(home["def_ppa"])
    )

    ppa_matchup_diff = (
        home_ppa_matchup
        -
        away_ppa_matchup
    )


    # --------------------------------------------------------
    # SUCCESS RATE MATCHUP
    # --------------------------------------------------------

    home_success_matchup = (
        float(home["off_success"])
        -
        float(away["def_success"])
    )

    away_success_matchup = (
        float(away["off_success"])
        -
        float(home["def_success"])
    )

    success_matchup_diff = (
        home_success_matchup
        -
        away_success_matchup
    )


    # --------------------------------------------------------
    # YARDS / PLAY MATCHUP
    # --------------------------------------------------------

    home_ypp_matchup = (
        float(home["off_ypp"])
        -
        float(away["def_ypp"])
    )

    away_ypp_matchup = (
        float(away["off_ypp"])
        -
        float(home["def_ypp"])
    )

    ypp_matchup_diff = (
        home_ypp_matchup
        -
        away_ypp_matchup
    )


    # --------------------------------------------------------
    # EXPLOSIVENESS MATCHUP
    # --------------------------------------------------------

    home_explosive_matchup = (
        float(home["off_explosive"])
        -
        float(away["def_explosive"])
    )

    away_explosive_matchup = (
        float(away["off_explosive"])
        -
        float(home["def_explosive"])
    )

    explosive_matchup_diff = (
        home_explosive_matchup
        -
        away_explosive_matchup
    )


    # --------------------------------------------------------
    # CURRENT PASS / RUSH MATCHUPS
    # --------------------------------------------------------

    split_values = calculate_live_split_features(
        away_team,
        home_team,
    )

    pass_matchup_diff = (
        split_values["pass_matchup_diff"]
    )

    rush_matchup_diff = (
        split_values["rush_matchup_diff"]
    )


    # --------------------------------------------------------
    # OPPONENT-ADJUSTED MATCHUPS
    # --------------------------------------------------------

    adj_off_matchup_diff = (
        home_adj["adj_off_ppa"]
        -
        away_adj["adj_off_ppa"]
    )

    # More-negative defensive PPA is better.
    # Preserve original Colab sign convention.

    adj_def_matchup_diff = (
        away_adj["adj_def_ppa"]
        -
        home_adj["adj_def_ppa"]
    )

    overall_adj_matchup = (
        adj_off_matchup_diff
        +
        adj_def_matchup_diff
    )


    # --------------------------------------------------------
    # FINAL 21-FEATURE ROW
    # --------------------------------------------------------

    row = pd.DataFrame([{

        "home_field":
            0.0 if neutral_site else 1.0,

        "home_ppg":
            float(home["ppg"]),

        "away_ppg":
            float(away["ppg"]),

        "home_papg":
            float(home["papg"]),

        "away_papg":
            float(away["papg"]),

        "win_pct_diff":
            win_pct_diff,

        "avg_margin_diff":
            avg_margin_diff,

        "scoring_matchup_diff":
            scoring_matchup_diff,

        "ppa_matchup_diff":
            ppa_matchup_diff,

        "success_matchup_diff":
            success_matchup_diff,

        "ypp_matchup_diff":
            ypp_matchup_diff,

        "explosive_matchup_diff":
            explosive_matchup_diff,

        "pass_matchup_diff":
            pass_matchup_diff,

        "rush_matchup_diff":
            rush_matchup_diff,

        "home_adj_off_ppa":
            home_adj["adj_off_ppa"],

        "away_adj_off_ppa":
            away_adj["adj_off_ppa"],

        "home_adj_def_ppa":
            home_adj["adj_def_ppa"],

        "away_adj_def_ppa":
            away_adj["adj_def_ppa"],

        "adj_off_matchup_diff":
            adj_off_matchup_diff,

        "adj_def_matchup_diff":
            adj_def_matchup_diff,

        "overall_adj_matchup":
            overall_adj_matchup,

    }])

    missing = [
        col
        for col in features
        if col not in row.columns
    ]

    if missing:
        raise RuntimeError(
            f"Feature builder is missing trained model features: "
            f"{missing}"
        )

    # IMPORTANT:
    # Full precision is preserved here.
    # We only round values later for website DISPLAY.

    row = row[features].copy()

    numeric_row = row.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if numeric_row.isna().any().any():

        bad = (
            numeric_row
            .columns[
                numeric_row.isna().any()
            ]
            .tolist()
        )

        raise RuntimeError(
            f"Invalid/non-numeric model inputs: {bad}"
        )

    return numeric_row


# ============================================================
# USER INTERFACE
# ============================================================

st.title(
    "🏈 2026 College Football Game Predictor"
)

st.caption(
    "21-feature margin model + current 2026 matchup profiles"
)


with st.form("predict_form"):

    c1, c2 = st.columns(2)

    with c1:

        away = st.selectbox(
            "Away team",
            team_list,
            index=(
                team_list.index("Indiana")
                if "Indiana" in team_list
                else 0
            ),
        )

    with c2:

        home = st.selectbox(
            "Home team",
            team_list,
            index=(
                team_list.index("Nebraska")
                if "Nebraska" in team_list
                else min(
                    1,
                    len(team_list) - 1,
                )
            ),
        )

    neutral = st.checkbox(
        "Neutral site",
        value=False,
    )

    st.markdown(
        "#### Optional sportsbook lines"
    )

    st.caption(
        "Enter the HOME team's line. "
        "Example: Nebraska +7.5 → enter 7.5"
    )

    c3, c4 = st.columns(2)

    with c3:

        market_spread_text = st.text_input(
            "Home line",
            placeholder="+7.5",
        )

    with c4:

        market_total_text = st.text_input(
            "O/U",
            placeholder="51.5",
        )

    submitted = st.form_submit_button(
        "RUN PREDICTION",
        use_container_width=True,
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    if away == home:

        st.warning(
            "Select two different teams."
        )

        st.stop()

    try:

        feature_row = build_live_game_features(
            away,
            home,
            neutral_site=neutral,
        )

        # FULL PRECISION MODEL PREDICTION

        margin = float(
            np.asarray(
                margin_model.predict(
                    feature_row
                )
            ).reshape(-1)[0]
        )


        # ----------------------------------------------------
        # TEAM PROFILES
        # ----------------------------------------------------

        away_profile = get_profile(away)
        home_profile = get_profile(home)

        away_ppg = float(
            away_profile["ppg"]
        )

        away_papg = float(
            away_profile["papg"]
        )

        home_ppg = float(
            home_profile["ppg"]
        )

        home_papg = float(
            home_profile["papg"]
        )


        # ----------------------------------------------------
        # PROJECTED TOTAL
        # ----------------------------------------------------

        raw_away = (
            away_ppg
            +
            home_papg
        ) / 2.0

        raw_home = (
            home_ppg
            +
            away_papg
        ) / 2.0

        projected_total = (
            raw_away
            +
            raw_home
        )


        # ----------------------------------------------------
        # PROJECTED SCORE
        # ----------------------------------------------------

        projected_home = max(
            0.0,
            (
                projected_total
                +
                margin
            ) / 2.0,
        )

        projected_away = max(
            0.0,
            (
                projected_total
                -
                margin
            ) / 2.0,
        )

        # DISPLAY TWO DECIMAL PLACES.
        # Underlying projected values remain full precision.

        final_away = f"{projected_away:.2f}"
        final_home = f"{projected_home:.2f}"


        # ----------------------------------------------------
        # HALFTIME ESTIMATE
        # ----------------------------------------------------

        first_half_share = 0.47

        projected_half_away = (
            projected_away
            *
            first_half_share
        )

        projected_half_home = (
            projected_home
            *
            first_half_share
        )

        half_away = (
            f"{projected_half_away:.2f}"
        )

        half_home = (
            f"{projected_half_home:.2f}"
        )


        # ----------------------------------------------------
        # WINNER
        # ----------------------------------------------------

        if margin > 0:

            winner = home

        elif margin < 0:

            winner = away

        else:

            winner = "PICK'EM"

        winner_text = (
            "PICK'EM"
            if margin == 0
            else
            f"{winner} by {abs(margin):.2f}"
        )


        # ====================================================
        # GAME CARD
        # ====================================================

        st.divider()

        st.subheader(
            f"{away} @ {home}"
            if not neutral
            else
            f"{away} vs {home} — Neutral Site"
        )


        # ----------------------------------------------------
        # HALFTIME / FINAL
        # ----------------------------------------------------

        h1, h2 = st.columns(2)

        with h1:

            st.markdown(
                "#### Projected halftime"
            )

            st.metric(
                away,
                half_away,
            )

            st.metric(
                home,
                half_home,
            )

            st.caption(
                "Estimate only; not a separately "
                "trained first-half model."
            )

        with h2:

            st.markdown(
                "#### Projected final"
            )

            st.metric(
                away,
                final_away,
            )

            st.metric(
                home,
                final_home,
            )


        # ----------------------------------------------------
        # MODEL RESULT
        # ----------------------------------------------------

        m1, m2 = st.columns(2)

        m1.metric(
            "Model result",
            winner_text,
        )

        m2.metric(
            "Projected total",
            f"{projected_total:.2f}",
        )


        # ====================================================
        # OPTIONAL MARKET COMPARISON
        # ====================================================

        market_spread = safe_number(
            market_spread_text
        )

        market_total = safe_number(
            market_total_text
        )

        if (
            market_spread is not None
            or
            market_total is not None
        ):

            st.markdown(
                "### Market comparison"
            )


        # ----------------------------------------------------
        # SPREAD
        # ----------------------------------------------------

        if market_spread is not None:

            model_home_line = -margin

            spread_edge = abs(
                market_spread
                -
                model_home_line
            )

            if model_home_line < market_spread:

                side = home

            elif model_home_line > market_spread:

                side = away

            else:

                side = "No side"

            s1, s2, s3 = st.columns(3)

            s1.metric(
                "Market home line",
                f"{market_spread:+.2f}",
            )

            s2.metric(
                "Model home line",
                f"{model_home_line:+.2f}",
            )

            s3.metric(
                "Spread difference",
                f"{spread_edge:.2f} pts",
            )

            st.write(
                f"**Model side:** {side} — "
                f"**{edge_grade(spread_edge)}**"
            )


        # ----------------------------------------------------
        # TOTAL
        # ----------------------------------------------------

        if market_total is not None:

            total_diff = (
                projected_total
                -
                market_total
            )

            total_edge = abs(
                total_diff
            )

            if total_diff > 0:

                total_side = (
                    f"OVER {market_total:.2f}"
                )

            elif total_diff < 0:

                total_side = (
                    f"UNDER {market_total:.2f}"
                )

            else:

                total_side = "No edge"

            t1, t2, t3 = st.columns(3)

            t1.metric(
                "Market total",
                f"{market_total:.2f}",
            )

            t2.metric(
                "Model total",
                f"{projected_total:.2f}",
            )

            t3.metric(
                "Total difference",
                f"{total_edge:.2f} pts",
            )

            st.write(
                f"**Model side:** {total_side} — "
                f"**{edge_grade(total_edge)}**"
            )


        # ====================================================
        # MATCHUP SIGNALS
        # ====================================================

        with st.expander(
            "View matchup signals"
        ):

            signal_features = [

                (
                    "Avg Margin",
                    "avg_margin_diff",
                ),

                (
                    "Scoring",
                    "scoring_matchup_diff",
                ),

                (
                    "PPA",
                    "ppa_matchup_diff",
                ),

                (
                    "Success Rate",
                    "success_matchup_diff",
                ),

                (
                    "Yards / Play",
                    "ypp_matchup_diff",
                ),

                (
                    "Explosiveness",
                    "explosive_matchup_diff",
                ),

                (
                    "Pass Matchup",
                    "pass_matchup_diff",
                ),

                (
                    "Rush Matchup",
                    "rush_matchup_diff",
                ),

                (
                    "Overall Adjusted",
                    "overall_adj_matchup",
                ),
            ]

            row = feature_row.iloc[0]

            # DISPLAY ONLY:
            # Round matchup signals to two decimals.
            # This does NOT affect the model prediction.

            signal_data = [

                {
                    "Signal": label,
                    "Value":
                        round(
                            float(row[col]),
                            2,
                        ),
                }

                for label, col
                in signal_features

                if col in row.index
            ]

            st.dataframe(
                pd.DataFrame(
                    signal_data
                ),
                hide_index=True,
                use_container_width=True,
            )


        # ====================================================
        # 2026 DATA AUDIT
        # ====================================================

        with st.expander(
            "2026 data audit"
        ):

            audit = pd.DataFrame([

                {
                    "Team":
                        away,

                    "PPG":
                        float(
                            away_profile["ppg"]
                        ),

                    "PAPG":
                        float(
                            away_profile["papg"]
                        ),

                    "Win %":
                        float(
                            away_profile["win_pct"]
                        ),

                    "Off PPA":
                        float(
                            away_profile["off_ppa"]
                        ),

                    "Def PPA":
                        float(
                            away_profile["def_ppa"]
                        ),
                },

                {
                    "Team":
                        home,

                    "PPG":
                        float(
                            home_profile["ppg"]
                        ),

                    "PAPG":
                        float(
                            home_profile["papg"]
                        ),

                    "Win %":
                        float(
                            home_profile["win_pct"]
                        ),

                    "Off PPA":
                        float(
                            home_profile["off_ppa"]
                        ),

                    "Def PPA":
                        float(
                            home_profile["def_ppa"]
                        ),
                },
            ])

            # DISPLAY ONLY:
            # Two-decimal formatting for audit data.

            numeric_cols = [
                "PPG",
                "PAPG",
                "Win %",
                "Off PPA",
                "Def PPA",
            ]

            audit[numeric_cols] = (
                audit[numeric_cols]
                .round(2)
            )

            st.dataframe(
                audit,
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                f"Loaded {len(CURRENT_PROFILES)} "
                f"exported profiles; "
                f"{len(team_list)} teams are "
                f"available in the predictor."
            )


        st.caption(
            "Clean model bundle loaded from "
            "the working Colab pipeline."
        )


    except Exception as exc:

        st.exception(exc)