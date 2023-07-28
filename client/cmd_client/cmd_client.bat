@REM curl -X POST -F "index=2" http://localhost:5000/change-model
curl -X GET -F "image=@images/11847.jpg" "http://localhost:5000/mask"

@REM curl -X GET -F "image=@images/11847.jpg" "http://localhost:5000/predict?x=186&y=847"