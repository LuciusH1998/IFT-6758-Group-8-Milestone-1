import streamlit as st
import requests
from client.game_client import GameClient


st.title("Hockey Visualization App")
    
## Sidebar
with st.sidebar:
    st.header("Model Registry")
    # inputs
    workspace = st.text_input("Workspace", value="IFT6758-2025-B08")
    model_name = st.selectbox(
                "Which model would you like to use?",
                ("logreg_distance", "logreg_angle", "logreg_distance_angle"),
                index=None,
                placeholder="Select model...",
                key="model_name"
            )
    if st.session_state["model_name"] in ["logreg_distance", "logreg_angle", "logreg_distance_angle"]:
        model_version = st.selectbox(
                "Which model version would you like to use?",
                tuple(f"v{i}" for i in range(8)) + ("latest",),
                index=None,
                placeholder="Select model version...",
                key="model_version"
            )
    else:
        model_version = st.selectbox(
                "Which model version would you like to use?",
                ("latest",),
                index=None,
                placeholder="Select model version...",
                key="model_version"
            )

    if st.button("Get model"):
        if not workspace or not model_name or not model_version:
            # Clear the history of the previous run
            for key in ["homeTeamId", "awayTeamId", "df", "new_df", "teams", "eventID"]:
                st.session_state.pop(key, None)
            st.error("Please fill out all fields before loading a model.")
            st.stop()

        try:
            # Call serving container to download model
            response = requests.post(
                # "http://serving:5000/download_registry_model",
                "http://127.0.0.1:5000/download_registry_model",
                json={
                    "model": model_name,
                    "version": model_version
                }
            )
            if response.status_code == 200:
                st.success(f" Model {model_name}:{model_version} loaded successfully!")

                # Choosing the correct features to use when making predictions
                if model_name == "logreg_distance":
                    features = ["distance_from_net"]
                elif model_name == "logreg_angle":
                    features = ["angle_from_net"]
                else:
                    features = ["distance_from_net", "angle_from_net"]
                st.session_state["features"] = features

            else:
                st.error(f" Failed to load model: {response.text}")
        except Exception as e:
            st.error(f" Error connecting to serving container: {e}")

        # Clear the history of the previous run
        for key in ["homeTeamId", "awayTeamId", "df", "new_df", "teams", "eventID"]:
            st.session_state.pop(key, None)


## Game ID input
with st.container():
    game_id = str(st.text_input("Game ID", key="game_id"))


## Game info and predictions
with st.container():
    if st.button("Ping game"):
        if game_id:
            gclient = GameClient(game_id)
            df = gclient.extract()
            st.session_state["homeTeamId"], st.session_state["awayTeamId"] = df["homeTeamId"].drop_duplicates().values[0], df["awayTeamId"].drop_duplicates().values[0]
            try:
                # Create test data
                st.session_state['df']=df.copy()
                st.session_state['df'].dropna(subset=st.session_state["features"], inplace=True)
                
                # Send to serving container
                response = requests.post(
                    # "http://serving:5000/predict",2023020006
                    "http://127.0.0.1:5000/predict",
                    json=st.session_state['df'][st.session_state["features"]].to_dict(orient="list")
                )
                
                if response.status_code == 200:
                    predictions = response.json()["predictions"]
                    st.session_state['df']["model_predictions"] = response.json()["predictions"]
                    st.session_state["teams"] = df[["homeTeamName", "awayTeamName"]].drop_duplicates().values[0]
                    st.success(f"Loaded game succesfully!")
                else:
                    st.error(f"Prediction failed: {response.text}")

            except Exception as e:
                st.error(f"Error: {e}")

        else:
            st.warning("Please enter a game ID.")

with st.container():
    if "teams" in st.session_state:
        st.subheader(f"Game {game_id}: {st.session_state["teams"][0]} vs. {st.session_state["teams"][1]}")
        eventID = st.selectbox(
            "Which event ID would you like to consider in the chosen game?",
            sorted(st.session_state['df']["eventId"]),
            index=None,
            placeholder="Select event ID...",
            key="eventID"
        )

    if st.session_state.get("eventID") is not None:
        st.session_state['df'] = st.session_state['df'].reset_index(drop=True)
        event_idx = st.session_state['df'].loc[st.session_state['df']["eventId"] == st.session_state["eventID"]].index[0]
        st.session_state["new_df"] = st.session_state['df'].iloc[:event_idx+1]
        data = st.session_state["new_df"]
        st.text(f"Period {data.iloc[event_idx]['periodDescriptor.number']} - {data.iloc[event_idx]['timeRemaining']} left")

        col1, col2 = st.columns(2)
        with col1:
            xG_home = data.loc[data["details.eventOwnerTeamId"] == st.session_state["homeTeamId"]]["model_predictions"].sum().round(2)
            goals_home = data.loc[data["details.eventOwnerTeamId"] == st.session_state["homeTeamId"]]["is_goal"].sum()
            st.metric(f"{st.session_state["teams"][0]} xG (actual)", f"{xG_home} ({goals_home})", str((xG_home-goals_home).round(2)), delta_color="off")

        with col2:
            xG_away = data.loc[data["details.eventOwnerTeamId"] == st.session_state["awayTeamId"]]["model_predictions"].sum().round(2)
            goals_away = data.loc[data["details.eventOwnerTeamId"] == st.session_state["awayTeamId"]]["is_goal"].sum()
            st.metric(f"{st.session_state["teams"][1]} xG (actual)", f"{xG_away} ({goals_away})", str((xG_away-goals_away).round(2)), delta_color="off")


## Data used for predictions
with st.container():
    if st.session_state.get("eventID") is not None:
        st.subheader("Data used for predictions (and predictions)")
        st.dataframe(st.session_state["new_df"][st.session_state["features"] + ["model_predictions"]], use_container_width=True)
    
