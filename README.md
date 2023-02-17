# Spotify Clustering

## Purpose
This script uses the spotipy API to allow users to read song features from a list of playlists and generate new playlists from those songs using K-Means Clustering. This implementation was modeled mostly off of this blog post: https://towardsdatascience.com/k-means-clustering-using-spotify-song-features-9eb7d53d105c


## How to Use
This is not set up as a package, you may need to manually set up your python package environment given the packages in spotify_clustering.py

From the repo root directory:

python ./spotify_clustering.py 'user' 'input_playlist_names_path'

Where 'user' is the spotify user account, and 'input_playlist_names_path' is the path to the csv file of playlist names (no header)

NOTE: The Spotipy API client requires that SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_PASSWORD are set in the os environment.

NOTE: This requires that you first create a spotify developer application: https://developer.spotify.com/dashboard/applications


## TODO
I have been struggling to get write access for my spotify account with the spotipy API, but ideally this codebase should be able to automatically upload the new clusters as playlists in spotipy.

Additionally, I want to make this codebase more flexible in the future wrt e.g. the clustering algorithm, any additional feature engineering such as dimnsionality reduction to improve the model, etc.

Also, really need to add unit tests...
