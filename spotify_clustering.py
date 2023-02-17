import os
import sys
import csv
from typing import List, Dict, Any
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans


def read_playlist_names(input_playlist_names_path: str) -> List[str]:
    """
    Reads playlist names from headerless csv and returns a list of playlists.
    """
    playlist_names = []
    with open(input_playlist_names_path) as f:
        playlist_reader = csv.reader(
            f,
        )
        for row in playlist_reader:
            playlist_names.append(*row)
    return playlist_names


def get_songs(
    playlist_names: List[str], user: str, sp: spotipy.client.Spotify
) -> List[Dict[str, Any]]:
    """
    Expects environment variables for SPOTIPY_CLIENT_ID and
    SPOTIPY_CLIENT_SECRET from a user-created spotipy developer app.

    Loops over all user playlists in playlist_names and adds all songs to a
    songs list, requesting from spotify in batches of 50 songs, where each
    song is a dict containing various features and metadata.
    """
    playlists = sp.user_playlists(user)
    playlist_ids = []
    for playlist in playlists["items"]:
        if playlist["name"] in playlist_names:
            playlist_ids.append(playlist["id"])
    songs = []
    for playlist_id in playlist_ids:
        offset = 0
        while True:
            content = sp.user_playlist_tracks(
                user=user, playlist_id=playlist_id, limit=50, offset=offset
            )
            songs += content["items"]
            if content["next"] is not None:
                offset += 50
            else:
                break
        offset = 0
    return songs


def extract_features(
    songs: List[Dict[str, Any]], sp: spotipy.client.Spotify
) -> pd.DataFrame:
    """
    Extracts the audio features from the songs list using the spotipy API and
    returns a pandas dataframe of the song features.
    """
    song_metadata = {}
    ids = []
    for i in songs:
        track = i["track"]
        ids.append(track["id"])
        song_metadata[track["id"]] = {
            "name": track["name"],
            "artists": [x["name"] for x in track["artists"]],
        }
    index = 0
    audio_features = []
    # NOTE: If there are duplicate tracks, len(songs) != len(song_metadata)
    while index < len(song_metadata.keys()):
        # NOTE: I have no idea why but I had one track with no id...
        audio_features += sp.audio_features(
            [x for x in ids[index:index + 50] if x is not None]
        )
        index += 50
    features_list = []
    for features in audio_features:
        features_list.append(
            [
                song_metadata[features["id"]]["name"],
                song_metadata[features["id"]]["artists"],
                features["energy"],
                features["liveness"],
                features["tempo"],
                features["speechiness"],
                features["mode"],
                features["uri"],
            ]
        )
    return pd.DataFrame(
        features_list,
        columns=[
            "name",
            "artists",
            "energy",
            "liveness",
            "tempo",
            "speechiness",
            "mode",
            "uri",
        ],
    )


def scale_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Scales the audio features and returns a new dataframe adding the scaled
    features columns.
    """
    x = features.loc[
        :, ~features.columns.isin(["name", "artists", "uri"])
    ].values
    min_max_scaler = preprocessing.MinMaxScaler()
    features_scaled = min_max_scaler.fit_transform(x)
    features_scaled = pd.DataFrame(
        features_scaled,
        columns=[
            "energy_scaled",
            "liveness_scaled",
            "tempo_scaled",
            "speechiness_scaled",
            "mode_scaled",
        ],
    )
    return features.loc[:, ["name", "artists", "uri"]].join(features_scaled)


def find_clusters(
    features: pd.DataFrame,
    features_scaled: pd.DataFrame,
    n_clusters: int,
    max_iter: int,
) -> pd.DataFrame:
    """
    Creates clusters using KMeans clustering from the scaled audio features
    and adds columns of the clusters to the original features dataframe.
    """
    kmeans = KMeans(
        init="k-means++", n_clusters=n_clusters, max_iter=max_iter
    ).fit(
        features_scaled.loc[
            :, ~features_scaled.columns.isin(["name", "artists", "uri"])
        ]
    )
    features["kmeans"] = kmeans.labels_
    return features


# NOTE: Can't get write access from API for some reason so this is untested
def create_playlists(
    user: str,
    cluster_names: List[str],
    sp: spotipy.client.Spotify,
    public: bool = False,
    create: bool = False,
) -> None:
    """
    Loop over clusters in cluster_names and create playlists from user account.
    """
    if create:
        for cluster in cluster_names:
            sp.user_playlist_create(user, cluster, public)


# NOTE: Can't get write access from API for some reason so this is untested
def add_tracks_to_playlists(
    user: str,
    cluster_playlist_ids: List[str],
    clusters_df: pd.DataFrame,
    sp: spotipy.client.Spotify,
) -> None:
    """
    For each cluster in cluster_playlist_ids, upload all tracks for that
    cluster to the user account.
    """
    kmean_idx = 0
    for cluster_playlist_id in cluster_playlist_ids:
        tracks = clusters_df.query("kmeans=={kmean_idx}")["uri"]
        sp.user_playlist_add_tracks(
            user, cluster_playlist_id, tracks, position=None
        )
        kmean_idx += 1


if __name__ == "__main__":
    auth_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(auth_manager=auth_manager)
    playlist_names = read_playlist_names(sys.argv[2])
    songs = get_songs(playlist_names, sys.argv[1], sp)
    features = extract_features(songs, sp)
    features.to_csv(
        f"{os.getcwd()}/user_data/spotify_features.csv", index=False
    )
    features_scaled = scale_features(features)
    features_clustered = find_clusters(
        features, features_scaled, len(playlist_names), 500
    )
    features.to_csv(
        f"{os.getcwd()}/user_data/spotify_clusters.csv", index=False
    )
