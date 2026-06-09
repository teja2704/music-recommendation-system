import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotify_config import (
    get_spotify_client_credentials,
    get_spotify_redirect_uri,
)

# Load Spotify settings from the process environment.
CLIENT_ID, CLIENT_SECRET = get_spotify_client_credentials()
REDIRECT_URI = get_spotify_redirect_uri()

# Set up authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope="user-read-playback-state,user-modify-playback-state,user-read-currently-playing,user-library-read,user-library-modify,user-top-read,user-read-recently-played"
))

# Test the connection
user = sp.current_user()
print(f"Authenticated as: {user['display_name']}")
