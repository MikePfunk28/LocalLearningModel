# mvp_mcconnell_analysis.py

# 1. Define Representative Data
POSITION_2016 = "The American people should have a voice in the selection of their next Supreme Court Justice. Therefore, this vacancy should not be filled until we have a new president."
ACTION_2016 = "Blocked hearings for Merrick Garland."
RATIONALE_TYPE_2016 = "Defer to election / People's voice"

POSITION_2020 = "We will vote to confirm Justice Barrett this year. The Senate has a constitutional obligation to advise and consent, and we are fulfilling that duty with President Trump's nominee. The precedent is that when the presidency and the Senate are held by the same party, the nomination is typically confirmed." # Slightly more nuanced for MVP
ACTION_2020 = "Proceeded with confirmation of Amy Coney Barrett."
RATIONALE_TYPE_2020 = "Senate duty to act / Same party control"

# 2. Analysis Logic
def analyze_stances(data_2016: dict, data_2020: dict):
    """
    Analyzes the stances from 2016 and 2020 and produces a qualitative summary
    and data for a potential graph.
    """

    qualitative_summary = (
        "--- Qualitative Analysis of Stances ---\n"
        "A significant inconsistency is observed in the stated rationales and actions regarding Supreme Court nominations:\n\n"
        f"In 2016 (Obama - Garland):\n"
        f"  Position Stated: \"{data_2016['position']}\"\n"
        f"  Action Taken: {data_2016['action']}\n"
        f"  The core argument was to let the American people decide via an upcoming presidential election, thereby delaying the nomination process.\n\n"
        f"In 2020 (Trump - Barrett):\n"
        f"  Position Stated: \"{data_2020['position']}\"\n"
        f"  Action Taken: {data_2020['action']}\n"
        f"  The argument shifted to the Senate's constitutional duty to 'advise and consent', and the historical precedent of confirming a nominee when the Presidency and Senate are controlled by the same party, even close to an election.\n\n"
        "Conclusion: The justification for action changed significantly depending on the political context, particularly which party controlled the Presidency and the Senate. The principle applied in 2016 (deferring to the election) was not applied in 2020."
    )

    # Illustrative scores: Higher score = 'more consistent' with own party's immediate goals / more assertive action
    # Lower score = 'less consistent' or more obstructive action based on stated principle for other party
    # This is purely for MVP illustration of "how far off" from a neutral principle.
    graph_data = [
        {
            "period": "2016 (Obama - Garland)",
            "stated_rationale_type": data_2016['rationale_type'],
            "action_taken": data_2016['action'],
            "consistency_score_illustrative": 2  # Illustrative: Low score for blocking based on "people's voice"
        },
        {
            "period": "2020 (Trump - Barrett)",
            "stated_rationale_type": data_2020['rationale_type'],
            "action_taken": data_2020['action'],
            "consistency_score_illustrative": 8  # Illustrative: High score for proceeding based on "Senate duty"
        }
    ]

    return qualitative_summary, graph_data

# 3. Output Generation
if __name__ == "__main__":
    data_2016_input = {
        "position": POSITION_2016,
        "action": ACTION_2016,
        "rationale_type": RATIONALE_TYPE_2016
    }
    data_2020_input = {
        "position": POSITION_2020,
        "action": ACTION_2020,
        "rationale_type": RATIONALE_TYPE_2020
    }

    summary, graph_output = analyze_stances(data_2016_input, data_2020_input)

    print(summary)

    print("\n--- Data Structure for Potential Graph ---")
    # Basic print, for a real app might use json.dumps for better formatting
    for item in graph_output:
        print(item)

    print("\n--- Potential Visualization Description ---")
    visualization_description = (
        "The 'graph_data' structure could be used to generate various visualizations. For example:\n"
        "1. A Bar Chart: Each 'period' (e.g., '2016 (Obama - Garland)') would be a category on the x-axis. "
        "The 'consistency_score_illustrative' could be represented by the height of the bars on the y-axis. "
        "Annotations or tooltips could display the 'stated_rationale_type' and 'action_taken' for each bar.\n"
        "2. A Timeline Chart: Events plotted along a timeline, with annotations detailing the position and action at each key point (2016 and 2020), perhaps color-coded by the illustrative score or action type.\n"
        "The 'consistency_score_illustrative' is a simplified, illustrative metric for this MVP to highlight the differing approaches rather than a rigorous, objective measure."
    )
    print(visualization_description)
