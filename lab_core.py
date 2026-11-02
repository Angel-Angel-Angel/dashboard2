from io import BytesIO, StringIO
import csv
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
REQUIRED = ['zip', 'state_id', 'state_name', 'county_name', 'city', 'population', 'density']
MEASURES = ['population', 'density']
NAVY, GOLD, CYAN = '#003366', '#B8860B', '#007E9D'
PLOT_CONFIG = {'displaylogo': False, 'showSendToCloud': False, 'responsive': True}
def normalize_identifiers(frame):
    """Validate ZIP and county identifiers without assuming FIPS is globally unique."""
    data = frame.copy()

    # Validate ZIP codes and preserve leading zeros.
    data["zip"] = data["zip"].astype("string").str.strip()

    if data["zip"].isna().any() or not data["zip"].str.fullmatch(r"\d{1,5}").all():
        raise ValueError("ZIP must contain 1–5 digits.")

    data["zip"] = data["zip"].str.zfill(5)

    if data["zip"].duplicated().any():
        raise ValueError("Duplicate ZIP codes found. Inspect the source.")

    # Validate state identifiers.
    data["state_id"] = data["state_id"].astype("string").str.strip().str.upper()
    data["state_name"] = data["state_name"].astype("string").str.strip()

    if data[["state_id", "state_name"]].isna().any().any():
        raise ValueError("State identifiers/names are missing.")

    if not data["state_id"].str.fullmatch(r"[A-Z]{2}").all():
        raise ValueError("state_id must be a two-letter abbreviation.")

    if data.groupby("state_id")["state_name"].nunique().gt(1).any():
        raise ValueError("A state ID has multiple state names.")

    # Validate county FIPS within each state, not globally.
    if "county_fips" in data.columns:
        data["county_fips"] = (
            data["county_fips"]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

        present = data["county_fips"].notna()

        if not data.loc[present, "county_fips"].str.fullmatch(r"\d{1,5}").all():
            raise ValueError("county_fips must contain 1–5 digits or be blank.")

        data.loc[present, "county_fips"] = (
            data.loc[present, "county_fips"].str.zfill(5)
        )

        conflicts = (
            data.loc[present]
            .groupby(["state_id", "county_fips"])["county_name"]
            .nunique()
        )

        if conflicts.gt(1).any():
            raise ValueError(
                "A state/county FIPS combination has conflicting county names. "
                "Inspect the source records."
            )

    return data


def load_uszips(csv_bytes):
    """Read a UTF-8 CSV and retain missing measures; never silently discard rows."""
    try:
        source = csv_bytes.decode("utf-8-sig")
        header = next(csv.reader(StringIO(source)))
        columns = [name.strip().lower() for name in header]
        if len(set(columns)) != len(columns):
            raise ValueError("Duplicate column names after trimming/lowercasing.")
        data = pd.read_csv(StringIO(source), dtype="string", keep_default_na=False)
    except (UnicodeDecodeError, StopIteration, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("Export a nonempty UTF-8, comma-delimited CSV with a header.") from exc
    data.columns = columns
    missing = sorted(set(REQUIRED) - set(data.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if data.empty:
        raise ValueError("The export has no data rows.")
    for field in data.columns:
        data[field] = data[field].str.strip().replace({"": pd.NA, "NULL": pd.NA, "\\N": pd.NA})
    data = normalize_identifiers(data)
    for field in MEASURES:
        original = data[field]
        values = pd.to_numeric(original, errors="coerce").astype("float64")
        bad = original.notna() & (values.isna() | ~np.isfinite(values) | values.lt(0))
        if bad.any():
            raise ValueError(f"{field}: {int(bad.sum())} invalid values. Inspect tokens/units; do not fill with zero.")
        data[field] = values
    audit = pd.DataFrame({"field": REQUIRED,
                          "present": [int(data[c].notna().sum()) for c in REQUIRED],
                          "missing": [int(data[c].isna().sum()) for c in REQUIRED]})
    audit["coverage_pct"] = (100 * audit["present"] / len(data)).round(1)
    data.attrs["county_key"] = (
    ["state_id", "county_fips"]
    if "county_fips" in data and data["county_fips"].notna().all()
    else ["state_id", "county_name"]
)
    return data, audit


def attach_geography(data, level):
    """Assign each ZIP to exactly one source-provided group, retaining unassigned rows."""
    out = data.copy()
    if level == "State":
        out["group_id"] = "state:" + out["state_id"]
        out["geography"] = out["state_name"] + " (" + out["state_id"] + ")"
    elif level == "County":
        names = out["county_name"].fillna("(Unassigned county)")
        default_key = "fips" if "county_fips" in out and out["county_fips"].notna().all() else "state_name"
        complete_fips = out.attrs.get("county_key", default_key) == "fips"
        if complete_fips:
            # Consistent display labels even if a row lacks a name but has a valid FIPS.
            labels = out.groupby("county_fips")["county_name"].first()
            names = out["county_fips"].map(labels).fillna("(Unassigned county)")
        # Use ONE key policy for the entire input, avoiding split groups.
        out["group_id"] = ("county:" + out["county_fips"] if complete_fips else
                           "county:" + out["state_id"] + ":" + names)
        out["geography"] = names + ", " + out["state_id"]
        if complete_fips:
            out["geography"] = out["geography"] + " [" + out["county_fips"] + "]"
    elif level == "City":
        names = out["city"].fillna("(Unassigned USPS city)")
        out["group_id"] = "city:" + out["state_id"] + ":" + names
        out["geography"] = names + ", " + out["state_id"]
    else:
        raise ValueError("Geography must be State, County, or City.")
    return out


def summarize_geography(data, level):
    """Summarize assigned ZIP records, not underlying population distributions."""
    tagged = attach_geography(data, level)
    groups = tagged.groupby(
        ["group_id", "geography"],
        observed=True,
        dropna=False
    )

    summary = groups.agg(
        zip_count=("zip", "size"),
        population_n=("population", "count"),
        population_mean=("population", "mean"),
        density_mean=("density", "mean"),
        density_n=("density", "count"),
        density_median=("density", "median"),
    )

    # Preserve missing values rather than converting all-missing groups to zero.
    summary["population_sum"] = groups["population"].sum(min_count=1)

    return (
        summary.reset_index()
        .sort_values("geography")
        .reset_index(drop=True)
    )


def polish(fig, title, height=480):
    """Consistent, readable academic styling for notebook and dashboard figures."""
    fig.update_layout(template="plotly_white", title=dict(text=title, x=0.02),
                      font=dict(family="Arial", size=15, color=NAVY),
                      height=height, margin=dict(l=25, r=25, t=100, b=65),
                      colorway=[NAVY, CYAN, GOLD], showlegend=False)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return polish(fig, "No chart for this selection", height=350)


def chart_bar(summary, top_n=15):
    """Rank observed assigned-ZIP population totals; use a zero baseline."""
    ranked = (summary.dropna(subset=["population_sum"])
              .sort_values(["population_sum", "geography"], ascending=[False, True]).head(top_n))
    if ranked.empty:
        return empty_figure("No observed population estimates in the selected groups.")
    shown = ranked.sort_values(["population_sum", "geography"], ascending=[True, False])
    fig = px.bar(shown, x="population_sum", y="geography", orientation="h",
                 labels={"population_sum": "Sum of available assigned-ZIP population estimates",
                         "geography": "Geographic group"},
                 hover_data={"zip_count": True, "population_n": True}, color_discrete_sequence=[NAVY])
    fig.update_xaxes(rangemode="tozero", tickformat=",")
    return polish(fig, f"Population of assigned ZIPs: {len(ranked)} of {len(summary)} groups",
                  height=max(430, min(850, 220 + 29 * len(ranked))))


def chart_scatter(summary):
    import plotly.express as px

    pairs = summary.dropna(
        subset=["population_sum", "density_mean"]
    ).copy()

    if pairs.empty:
        return empty_figure(
            "No groups have both population and density summaries."
        )

    fig = px.scatter(
        pairs,
        x="population_sum",
        y="density_mean",
        hover_name="geography",
        hover_data=["zip_count", "population_n", "density_n"],
        labels={
            "population_sum": "Summed ZIP-level population",
            "density_mean": "Mean ZIP-level density",
        },
    )
    fig.update_xaxes(title="Summed ZIP-level population", tickformat=",")
    fig.update_yaxes(title="Mean ZIP-level density")
    return polish(
        fig,
        f"Population vs. density: {len(pairs)} geographic groups"
    )


def chart_box(study, level, group_ids):
    import plotly.graph_objects as go

    selected = study.loc[study["group_id"].isin(group_ids)].copy()
    groups = (
        selected[["group_id", "geography"]]
        .drop_duplicates()
        .sort_values("geography")
    )

    fig = go.Figure()

    for row in groups.itertuples(index=False):
        values = selected.loc[
            selected["group_id"].eq(row.group_id), "population"
        ].dropna()

        if not values.empty:
            fig.add_trace(
                go.Box(
                    x=values,
                    name=row.geography,
                    boxpoints="outliers",
                    orientation="h",
                )
            )

    if not fig.data:
        return empty_figure(
            "No observed ZIP-level population values in the selected groups."
        )

    fig.update_layout(
        title=f"ZIP-level population distributions by {level.lower()}",
        xaxis_title="Population per ZIP",
        yaxis_title="Geographic group",
        height=550,
    )
    fig.update_xaxes(tickformat=",")
    return polish(fig, f"Population distributions across {len(fig.data)} groups")
