import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Replace these with your Spotify App credentials
CLIENT_ID = "a20569cb114a4cd380f6c240f0ad744c"
CLIENT_SECRET = "76b780a37b444824a6372dfec6ec41f3"
REDIRECT_URI = "http://localhost:8888/callback"

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
