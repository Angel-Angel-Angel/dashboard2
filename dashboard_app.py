
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from lab_core import (
    attach_geography,
    chart_bar,
    chart_box,
    chart_scatter,
    load_uszips,
    summarize_geography,
)


def run_dashboard():
    """Render the U.S. ZIP data dashboard."""

    st.set_page_config(
        page_title="CPS5745 | U.S. ZIP Data Explorer",
        layout="wide",
    )

    st.markdown(
        '<div style="border-bottom:4px solid #C4A45A;color:#002D5B;'
        'padding-bottom:0.5rem;font-weight:700">'
        'KEAN UNIVERSITY · CPS5745 · FALL 2026</div>',
        unsafe_allow_html=True,
    )

    st.title("U.S. ZIP Data Explorer")
    st.caption("Interactive Information Visualization · Julie (Yulia) Kumar")

    @st.cache_data(show_spinner=False, max_entries=4)
    def read_data(csv_bytes):
        return load_uszips(csv_bytes)

    data_folder = Path(__file__).resolve().parent / "data"

    with st.sidebar:
        st.header("1 · Choose the project data")
        uploaded = st.file_uploader(
            "Upload uszips.csv",
            type=["csv"],
            key="csv_upload",
        )
        st.caption(
            "An upload replaces the local data/uszips.csv for this session."
        )

    metadata = {}

    try:
        if uploaded is not None:
            raw = uploaded.getvalue()
            filename = uploaded.name

        elif (data_folder / "uszips.csv").is_file():
            raw = (data_folder / "uszips.csv").read_bytes()
            filename = "uszips.csv"

            metadata_path = data_folder / "source_metadata.json"

            if metadata_path.is_file():
                try:
                    metadata = json.loads(
                        metadata_path.read_text(encoding="utf-8")
                    )
                    if not isinstance(metadata, dict):
                        raise ValueError("Metadata must be a JSON object.")
                except (OSError, ValueError, UnicodeDecodeError):
                    st.warning(
                        "Source metadata could not be read. "
                        "Enter its source and vintage below."
                    )
                    metadata = {}

        else:
            st.info(
                "Upload the project's CSV in the sidebar, "
                "or place it at data/uszips.csv."
            )
            st.write(
                "The dashboard will use your project file. "
                "No example records are substituted."
            )
            st.code(
                "zip, state_id, state_name, county_name, city, "
                "population, density",
                language="text",
            )
            st.caption(
                "Required columns include ZIP, state, county, city, "
                "population, and density. Keep ZIP codes as text."
            )
            st.stop()

        df, audit = read_data(raw)

    except (
        OSError,
        ValueError,
        UnicodeDecodeError,
        pd.errors.ParserError,
    ) as exc:
        st.error(f"The CSV could not be used: {exc}")
        st.info(
            "Check that the CSV contains the required ZIP, geography, "
            "population, and density columns."
        )
        st.stop()

    digest = sha256(raw).hexdigest()

    with st.sidebar:
        source = st.text_input(
            "Data source / provider",
            value=str(metadata.get("source", "")),
            key=f"source_{digest}",
        )
        vintage = st.text_input(
            "Release / survey years",
            value=str(metadata.get("vintage", "")),
            key=f"vintage_{digest}",
        )

    st.caption(f"Loaded file: {filename}")
    st.caption(f"SHA-256: {digest}")
    st.caption("Validation covers the entire input file before filtering.")
    st.dataframe(audit, hide_index=True, use_container_width=True)

    st.info(
        "Population is the sum of observed ZIP-level population values. "
        "Density is summarized as the unweighted mean of available "
        "ZIP-level density values. These statistics describe the selected "
        "ZIP records, not necessarily the underlying population distribution "
        "of an entire county or city. ZIPs can cross geographic boundaries, "
        "so each row follows its assigned primary geography. Check data "
        "coverage and source years when interpreting the results."
    )

    with st.sidebar:
        st.header("2 · Filter every view")

        saved = metadata.get("initial_filters", {})
        saved = saved if isinstance(saved, dict) else {}

        levels = ["State", "County", "City"]
        start_level = saved.get("level", "State")

        level = st.selectbox(
            "Group ZIP rows by",
            levels,
            index=levels.index(start_level) if start_level in levels else 0,
            key="level",
        )

        states = sorted(df["state_id"].dropna().unique().tolist())
        start_states = saved.get("states")
        start_states = (
            states
            if start_states is None
            else [state for state in start_states if state in states]
        )

        chosen_states = st.multiselect(
            "States / territories",
            states,
            default=start_states,
            key=f"states_{digest}",
        )

        st.caption("Clearing this selection produces an empty result.")

        min_zip = st.number_input(
            "Minimum ZIP rows per group",
            min_value=1,
            value=max(1, int(saved.get("minimum", 1))),
            step=1,
            key="min_zip",
        )

        st.header("3 · Limit chart labels")
        start_top = max(1, int(saved.get("top", 15)))

        top_n = st.slider(
            "Top N groups in bar and box views",
            1,
            max(30, start_top),
            start_top,
            1,
            key="top_n",
        )

        st.caption(
            "Top N changes the bar and box displays. "
            "The scatterplot and summary table use all filtered groups."
        )

    # Apply filters consistently across all views.
    selected_states = df.loc[
        df["state_id"].isin(chosen_states)
    ].copy()

    grouped = attach_geography(selected_states, level)
    summary = summarize_geography(selected_states, level)

    summary = summary.loc[
        summary["zip_count"] >= int(min_zip)
    ].copy()

    selected = grouped.loc[
        grouped["group_id"].isin(summary["group_id"])
    ].copy()

    population_sum = selected["population"].sum(min_count=1)

    col1, col2, col3 = st.columns(3)
    col1.metric("Selected ZIP rows", f"{len(selected):,}")
    col2.metric(f"Selected {level.lower()} groups", f"{len(summary):,}")
    col3.metric(
        "Observed population sum",
        "Missing"
        if pd.isna(population_sum)
        else f"{population_sum:,.0f}",
    )

    observed = {
        "population": "Population",
        "density": "Density",
    }

    st.caption(
        "Observed ZIP rows: "
        + " · ".join(
            f"{label} {selected[field].notna().sum():,}/{len(selected):,}"
            for field, label in observed.items()
        )
    )

    if selected.empty:
        st.info(
            "No ZIP rows match these filters. Select at least one state "
            "or reduce the minimum group size."
        )
        st.stop()

    # Keep missing population sums last; break ties alphabetically.
    ranked = (
        summary.sort_values(
            ["population_sum", "geography"],
            ascending=[False, True],
            na_position="last",
        )
        .head(int(top_n))
    )

    plot_config = {
        "displaylogo": False,
        "showSendToCloud": False,
    }

    # View 1: population bar chart.
    st.subheader("View 1 · Compare ZIP population totals")
    st.caption(
        f"Up to {top_n} groups, ranked by the sum of available "
        "ZIP-level population values. Missing values are not zero."
    )

    if summary["population_sum"].notna().any():
        st.plotly_chart(
            chart_bar(summary, top_n=int(top_n)),
            use_container_width=True,
            theme=None,
            config=plot_config,
            key="population_bar",
        )
    else:
        st.info(
            "Population is missing for every selected ZIP row, "
            "so population ranking is unavailable."
        )

    # View 2: population versus density scatterplot.
    st.subheader("View 2 · Explore population and density")

    paired = summary[
        ["population_sum", "density_mean"]
    ].notna().all(axis=1).sum()

    st.caption(
        f"{paired:,}/{len(summary):,} filtered groups have both a "
        "population sum and a mean ZIP-level density and can appear as points."
    )

    if paired:
        st.plotly_chart(
            chart_scatter(summary),
            use_container_width=True,
            theme=None,
            config=plot_config,
            key="population_density_scatter",
        )

        if paired < 2:
            st.info(
                "Only one group has both measures. Include more groups "
                "to compare their positions."
            )
    else:
        st.info(
            "No selected group has both population and density summaries."
        )

    # View 3: population distribution boxplots.
    st.subheader("View 3 · Inspect ZIP-level population variation")

    st.caption(
        "Box plots show the distribution of individual ZIP-level population "
        "values within the displayed groups. Groups follow the population "
        "ranking; groups with missing population follow alphabetically. "
        "Each ZIP row receives equal weight."
    )

    box_rows = selected.loc[
        selected["group_id"].isin(ranked["group_id"])
    ]

    if box_rows["population"].notna().any():
        st.plotly_chart(
            chart_box(
                selected,
                level,
                group_ids=ranked["group_id"].tolist(),
            ),
            use_container_width=True,
            theme=None,
            config=plot_config,
            key="population_box",
        )
    else:
        st.info(
            "No observed ZIP-level population values are available "
            "for these displayed groups."
        )

    # Tables and downloads.
    st.subheader("Inspect and download the filtered data")
    st.caption(
        "These tables include all groups that pass the filters, regardless "
        "of Top N. When opening the ZIP CSV in a spreadsheet, import the "
        "zip column as text."
    )

    tabs = st.tabs(["Geography summaries", "ZIP rows"])
    tables = [summary, selected]
    names = ["geography_summary", "selected_zip_rows"]

    for tab, table, name in zip(tabs, tables, names):
        with tab:
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.download_button(
                "Download CSV",
                table.to_csv(index=False).encode("utf-8"),
                file_name=f"{name}.csv",
                mime="text/csv",
                key=f"download_{name}",
            )


if __name__ == "__main__":
    run_dashboard()

