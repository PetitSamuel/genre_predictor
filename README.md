# Genre Predictor
Get a Spotify API key at https://developer.spotify.com/documentation/web-api/quick-start/

Create a file called `.env` and add the following values:
```
SPOTIFY_CLIENT_ID =<your client id>
SPOTIFY_CLIENT_SECRET =<your client secret>
```

Install:
```
pip install -r requirements.txt 
```

Then run and it should work


## TODO   
[x] Query songs form a provided playlist ID    
[x] Query audio features from a song id    
[x] Query artist genre from an artist id    
[x] Map genres into a manageable sized set (from thousands into ~ 20)    
[x] Merge relevant data together    
[x] Plot our basic data (density graphs & genre counts)    
[] Test a baseline model (WIP: added a sample model)    
[] identify a set of models to experiment with    
[] select which ones to test in depth    
[] feature engineering: select which features to use for which model & which features to modify    
