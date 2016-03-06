export GOPATH="/home/ubuntu/euphoric-gpu"
cd /home/ubuntu/euphoric-gpu/src/slave
(go run slave.go 107.20.178.226) > ../../output.log 2>&1 &
